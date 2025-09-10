use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use futures::StreamExt;
use futures::future::join_all;
use arrow::compute::prep_null_mask_filter;
// Note: chrono is not available, using std::time::SystemTime instead
use clap::{Parser};
use arrow::array::{Array,  BinaryArray,  StringArray};
use arrow::datatypes::{SchemaRef};
use datafusion::datasource::physical_plan::parquet::{build_row_filter, DefaultParquetFileReaderFactory, PagePruningAccessPlanFilter, ParquetAccessPlan, RowGroupAccessPlanFilter};
use datafusion::datasource::physical_plan::{FileMeta, ParquetFileMetrics, ParquetFileReaderFactory};
use datafusion::datasource::schema_adapter::{DefaultSchemaAdapterFactory, SchemaAdapterFactory};
use datafusion::error::DataFusionError;
use datafusion::parquet::arrow::arrow_reader::{ArrowPredicate, ArrowReaderMetadata, ParquetRecordBatchReader, RowFilter, RowSelection, RowSelector};
use datafusion::parquet::arrow::async_reader::{AsyncFileReader, InMemoryRowGroup};
use datafusion::parquet::arrow::array_reader::RowGroups;
use datafusion::parquet::file::metadata::ParquetMetaData;
use datafusion::parquet::schema::types::{SchemaDescriptor};
use log::debug;
use object_store::local::LocalFileSystem;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Operator};
use datafusion::parquet::arrow::{arrow_reader::ArrowReaderBuilder, arrow_reader::ArrowReaderOptions};
use datafusion::parquet::arrow::{parquet_to_arrow_field_levels, FieldLevels, ProjectionMask};
use datafusion::common::{Column, Result, ScalarValue};
use datafusion::physical_expr::planner::logical2physical;
use datafusion::physical_optimizer::pruning::{build_pruning_predicate, PruningPredicate};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder};
use datafusion::physical_plan::{PhysicalExpr};
use datafusion::physical_plan::metrics::Count;
use datafusion::prelude::*;
use datafusion::logical_expr::BinaryExpr;

use object_store::ObjectMeta;
use promql_parser::label::{MatchOp, Matcher, Matchers};
use regex::Regex;

#[derive(Parser)]
#[command(version)]
/// Searches and materializes data from Parquet files with synthetic series data, while logging timing and memory usage.
struct Args {
    #[arg(long)]
    /// Optional matcher string in format "label_name=label_value" or "label_name!=label_value" or "label_name=~regex"
    /// Multiple matchers can be separated by commas: "label1=value1,label2!=value2,label3=~regex"
    /// If not provided, all rows will be materialized
    matcher: Option<String>,

    #[arg(long)]
    /// Comma-separated list of column names to project/materialize. If not specified, 
    /// all columns will be materialized based on column indexes. Example: "col1,col2,col3"
    projection: Option<String>,

    /// Path to the file to read
    path: PathBuf,
}


pub struct ReadPlanBuilder {
    batch_size: usize,
    /// Current to apply, includes all filters
    selection: Option<RowSelection>,
}

impl ReadPlanBuilder {
    /// Create a `ReadPlanBuilder` with the given batch size
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            selection: None,
        }
    }

    /// Set the current selection to the given value
    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    /// Returns the current selection, if any
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.as_ref()
    }

    /// Returns true if the current plan selects any rows
    pub fn selects_any(&self) -> bool {
        self.selection
            .as_ref()
            .map(|s| s.selects_any())
            .unwrap_or(true)
    }

    /// Evaluates an [`ArrowPredicate`], updating this plan's `selection`
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `predicate`.
    ///
    /// Note: pre-existing selections may come from evaluating a previous predicate
    /// or if the [`ParquetRecordBatchReader`] specified an explicit
    /// [`RowSelection`] in addition to one or more predicates.
    pub fn with_predicate(
        mut self,
        levels: &FieldLevels,
        row_groups: &dyn RowGroups,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        let mut reader = ParquetRecordBatchReader::try_new_with_row_groups(levels, row_groups, self.batch_size, self.selection.clone(), predicate.projection().clone())?;
        // let reader = ParquetRecordBatchReader::new(array_reader, self.clone().build());
        let mut filters = vec![];
        while let Some(maybe_batch) = reader.next() {
            let maybe_batch = maybe_batch?;
            let input_rows = maybe_batch.num_rows();
            let filter = predicate.evaluate(maybe_batch)?;
            // Since user supplied predicate, check error here to catch bugs quickly
            if filter.len() != input_rows {
                return Err(DataFusionError::ArrowError(
                    Box::new(arrow::error::ArrowError::InvalidArgumentError(
                        format!("ArrowPredicate predicate returned {} rows, expected {}", filter.len(), input_rows)
                    )),
                    None
                ));
            }
            match filter.null_count() {
                0 => filters.push(filter),
                _ => filters.push(prep_null_mask_filter(&filter)),
            };
        }

        let raw = RowSelection::from_filters(&filters);
        self.selection = match self.selection.take() {
            Some(selection) => Some(selection.and_then(&raw)),
            None => Some(raw),
        };
        Ok(self)
    }
}

/// Parse a matcher string into Matchers
/// Supports formats: "label_name=value", "label_name!=value", "label_name=~regex", "label_name!~regex"
/// Multiple matchers can be separated by commas: "label1=value1,label2!=value2"
fn parse_matcher_string(matcher_str: &str) -> Result<Matchers> {
    let matcher_str = matcher_str.trim();
    
    // Split by commas to handle multiple matchers
    let matcher_parts: Vec<&str> = matcher_str.split(',').map(|s| s.trim()).collect();
    let mut matchers = Vec::new();
    
    for part in matcher_parts {
        if part.is_empty() {
            continue; // Skip empty parts
        }
        
        // Parse individual matcher
        let (label_name, match_op, value) = parse_single_matcher(part)?;
        let matcher = Matcher::new(match_op, label_name.trim(), value.trim());
        matchers.push(matcher);
    }
    
    if matchers.is_empty() {
        return Err(DataFusionError::ArrowError(
            Box::new(arrow::error::ArrowError::InvalidArgumentError(
                format!("No valid matchers found in: {}", matcher_str)
            )),
            None
        ));
    }
    
    Ok(Matchers::new(matchers))
}

/// Parse a projection string into a vector of column names
fn parse_projection_string(projection_str: &str) -> Result<Vec<String>> {
    let projection_str = projection_str.trim();
    
    if projection_str.is_empty() {
        return Ok(vec![]);
    }
    
    // Split by commas to handle multiple columns
    let columns: Vec<String> = projection_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    if columns.is_empty() {
        return Err(DataFusionError::ArrowError(
            Box::new(arrow::error::ArrowError::InvalidArgumentError(
                format!("No valid columns found in projection: {}", projection_str)
            )),
            None
        ));
    }
    
    Ok(columns)
}

/// Parse a single matcher string into its components
fn parse_single_matcher(matcher_str: &str) -> Result<(String, MatchOp, String)> {
    let matcher_str = matcher_str.trim();
    
    // Parse different match operators
    if matcher_str.contains("!~") {
        let parts: Vec<&str> = matcher_str.split("!~").collect();
        if parts.len() != 2 {
            return Err(DataFusionError::ArrowError(
                Box::new(arrow::error::ArrowError::InvalidArgumentError(
                    format!("Invalid matcher format: {}. Expected 'label!~regex'", matcher_str)
                )),
                None
            ));
        }
        let regex = Regex::new(parts[1]).map_err(|e| DataFusionError::ArrowError(
            Box::new(arrow::error::ArrowError::InvalidArgumentError(
                format!("Invalid regex pattern '{}': {}", parts[1], e)
            )),
            None
        ))?;
        Ok((parts[0].to_string(), MatchOp::NotRe(regex), parts[1].to_string()))
    } else if matcher_str.contains("=~") {
        let parts: Vec<&str> = matcher_str.split("=~").collect();
        if parts.len() != 2 {
            return Err(DataFusionError::ArrowError(
                Box::new(arrow::error::ArrowError::InvalidArgumentError(
                    format!("Invalid matcher format: {}. Expected 'label=~regex'", matcher_str)
                )),
                None
            ));
        }
        let regex = Regex::new(parts[1]).map_err(|e| DataFusionError::ArrowError(
            Box::new(arrow::error::ArrowError::InvalidArgumentError(
                format!("Invalid regex pattern '{}': {}", parts[1], e)
            )),
            None
        ))?;
        Ok((parts[0].to_string(), MatchOp::Re(regex), parts[1].to_string()))
    } else if matcher_str.contains("!=") {
        let parts: Vec<&str> = matcher_str.split("!=").collect();
        if parts.len() != 2 {
            return Err(DataFusionError::ArrowError(
                Box::new(arrow::error::ArrowError::InvalidArgumentError(
                    format!("Invalid matcher format: {}. Expected 'label!=value'", matcher_str)
                )),
                None
            ));
        }
        Ok((parts[0].to_string(), MatchOp::NotEqual, parts[1].to_string()))
    } else if matcher_str.contains("=") {
        let parts: Vec<&str> = matcher_str.split("=").collect();
        if parts.len() != 2 {
            return Err(DataFusionError::ArrowError(
                Box::new(arrow::error::ArrowError::InvalidArgumentError(
                    format!("Invalid matcher format: {}. Expected 'label=value'", matcher_str)
                )),
                None
            ));
        }
        Ok((parts[0].to_string(), MatchOp::Equal, parts[1].to_string()))
    } else {
        return Err(DataFusionError::ArrowError(
            Box::new(arrow::error::ArrowError::InvalidArgumentError(
                format!("Invalid matcher format: {}. Expected 'label=value', 'label!=value', 'label=~regex', or 'label!~regex'", matcher_str)
            )),
            None
        ));
    }
}

pub struct SearchReaderBuilder {
    input: Box<dyn AsyncFileReader>,

    pub metadata: Arc<ParquetMetaData>,

    pub schema: SchemaRef,

    pub batch_size: usize,

    pub row_groups: Option<Vec<usize>>,

    pub filter: Option<RowFilter>,

    pub selection: Option<RowSelection>,
}

pub struct SearchReader {
    metadata: Arc<ParquetMetaData>,

    input: Box<dyn AsyncFileReader>,

    row_groups: VecDeque<usize>,

    selection: Option<RowSelection>,

    /// Optional filter
    filter: Option<RowFilter>,
}

impl SearchReaderBuilder {
    pub fn new_builder(input: Box<dyn AsyncFileReader>, metadata: ArrowReaderMetadata) -> Self {
        Self {
            input,
            metadata: metadata.metadata().clone(),
            schema: metadata.schema().clone(),
            batch_size: 1024,
            row_groups: None,
            filter: None,
            selection: None,
        }
    }

    /// Returns a reference to the [`ParquetMetaData`] for this parquet file
    pub fn metadata(&self) -> &Arc<ParquetMetaData> {
        &self.metadata
    }

    /// Returns the parquet [`SchemaDescriptor`] for this parquet file
    pub fn parquet_schema(&self) -> &SchemaDescriptor {
        self.metadata.file_metadata().schema_descr()
    }

    /// Returns the arrow [`SchemaRef`] for this parquet file
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Set the size of [`RecordBatch`] to produce. Defaults to 1024
    /// If the batch_size more than the file row count, use the file row count.
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        // Try to avoid allocate large buffer
        let batch_size = batch_size.min(self.metadata.file_metadata().num_rows() as usize);
        Self { batch_size, ..self }
    }

    /// Only read data from the provided row group indexes
    ///
    /// This is also called row group filtering
    pub fn with_row_groups(self, row_groups: Vec<usize>) -> Self {
        Self {
            row_groups: Some(row_groups),
            ..self
        }
    }

    /// Provide a [`RowSelection`] to filter out rows, and avoid fetching their
    /// data into memory.
    ///
    /// This feature is used to restrict which rows are decoded within row
    /// groups, skipping ranges of rows that are not needed. Such selections
    /// could be determined by evaluating predicates against the parquet page
    /// [`Index`] or some other external information available to a query
    /// engine.
    ///
    /// # Notes
    ///
    /// Row group filtering (see [`Self::with_row_groups`]) is applied prior to
    /// applying the row selection, and therefore rows from skipped row groups
    /// should not be included in the [`RowSelection`] (see example below)
    ///
    /// It is recommended to enable writing the page index if using this
    /// functionality, to allow more efficient skipping over data pages. See
    /// [`ArrowReaderOptions::with_page_index`].
    ///
    /// # Example
    ///
    /// Given a parquet file with 4 row groups, and a row group filter of `[0,
    /// 2, 3]`, in order to scan rows 50-100 in row group 2 and rows 200-300 in
    /// row group 3:
    ///
    /// ```text
    ///   Row Group 0, 1000 rows (selected)
    ///   Row Group 1, 1000 rows (skipped)
    ///   Row Group 2, 1000 rows (selected, but want to only scan rows 50-100)
    ///   Row Group 3, 1000 rows (selected, but want to only scan rows 200-300)
    /// ```
    ///
    /// You could pass the following [`RowSelection`]:
    ///
    /// ```text
    ///  Select 1000    (scan all rows in row group 0)
    ///  Skip 50        (skip the first 50 rows in row group 2)
    ///  Select 50      (scan rows 50-100 in row group 2)
    ///  Skip 900       (skip the remaining rows in row group 2)
    ///  Skip 200       (skip the first 200 rows in row group 3)
    ///  Select 100     (scan rows 200-300 in row group 3)
    ///  Skip 700       (skip the remaining rows in row group 3)
    /// ```
    /// Note there is no entry for the (entirely) skipped row group 1.
    ///
    /// Note you can represent the same selection with fewer entries. Instead of
    ///
    /// ```text
    ///  Skip 900       (skip the remaining rows in row group 2)
    ///  Skip 200       (skip the first 200 rows in row group 3)
    /// ```
    ///
    /// you could use
    ///
    /// ```text
    /// Skip 1100      (skip the remaining 900 rows in row group 2 and the first 200 rows in row group 3)
    /// ```
    ///
    /// [`Index`]: crate::file::page_index::index::Index
    pub fn with_row_selection(self, selection: RowSelection) -> Self {
        Self {
            selection: Some(selection),
            ..self
        }
    }

    /// Provide a [`RowFilter`] to skip decoding rows
    ///
    /// Row filters are applied after row group selection and row selection
    ///
    /// It is recommended to enable reading the page index if using this functionality, to allow
    /// more efficient skipping over data pages. See [`ArrowReaderOptions::with_page_index`].
    pub fn with_row_filter(self, filter: RowFilter) -> Self {
        Self {
            filter: Some(filter),
            ..self
        }
    }

    fn build(self) -> Result<SearchReader> {
        let num_row_groups = self.metadata.row_groups().len();

        let row_groups = match self.row_groups {
            Some(row_groups) => {
                if let Some(col) = row_groups.iter().find(|x| **x >= num_row_groups) {
                    return Err(DataFusionError::ArrowError(
                        Box::new(arrow::error::ArrowError::InvalidArgumentError(
                            format!("row group {} out of bounds 0..{}", col, num_row_groups)
                        )),
                        None
                    ));
                }
                row_groups.into()
            }
            None => (0..self.metadata.row_groups().len()).collect(),
        };

        // Try to avoid allocate large buffer
        let _batch_size = self
            .batch_size
            .min(self.metadata.file_metadata().num_rows() as usize);
        
        Ok(SearchReader {
            metadata: self.metadata,
            row_groups,
            selection: self.selection,
            input: self.input,
            filter: self.filter,
        })
    }
}

impl SearchReader {
    /// Returns a reference to the row groups to be read
    pub fn row_groups(&self) -> &VecDeque<usize> {
        &self.row_groups
    }

    /// Reads the next row group with the provided `selection` and `batch_size`
    ///
    /// Note: this captures self so that the resulting future has a static lifetime
    pub async fn read_row_group(
        mut self,
        row_group_idx: usize,
        selection: Option<RowSelection>,
        batch_size: usize,
    ) -> Result<Option<RowSelection>> {
        // TODO: calling build_array multiple times is wasteful
        let meta = self.metadata.row_group(row_group_idx);
        let offset_index = self
            .metadata
            .offset_index()
            // filter out empty offset indexes (old versions specified Some(vec![]) when no present)
            .filter(|index| !index.is_empty())
            .map(|x| x[row_group_idx].as_slice());

        let mut row_group = InMemoryRowGroup::new(
            offset_index,
            vec![None; meta.num_columns()],
            meta.num_rows() as usize,
            row_group_idx,
            self.metadata.as_ref(),
        );

        let filter = self.filter.as_mut();
        let mut plan_builder = ReadPlanBuilder::new(batch_size).with_selection(selection);

        // Update selection based on any filters
        if let Some(filter) = filter {
            for predicate in filter.predicates_mut() {
                if !plan_builder.selects_any() {
                    return Ok(None); // ruled out entire row group
                }

                // (pre) Fetch only the columns that are selected by the predicate
                let selection = plan_builder.selection();
                row_group
                    .fetch(&mut self.input, predicate.projection(), selection, batch_size, None)
                    .await?;

                let levels = parquet_to_arrow_field_levels(
                    &self.metadata.file_metadata().schema_descr_ptr(),
                    predicate.projection().clone(),
                    None,
                )?;

                plan_builder = plan_builder.with_predicate(&levels, &row_group, &mut **predicate)?;
            }
        }

        Ok(plan_builder.selection().cloned())
    }
}

fn decode_int_slice(data: &[u8]) -> Result<Vec<u64>> {
    let mut result = Vec::new();
    let mut pos = 0;
    if pos >= data.len() {
        return Ok(result);
    }
    let mut len = 0u64;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            return Err(DataFusionError::Internal("Unexpected end of data while decoding length".to_string()));
        }
        let byte = data[pos];
        pos += 1;
        len |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
    }
    for _ in 0..len {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            if pos >= data.len() {
                return Err(DataFusionError::Internal("Unexpected end of data while decoding value".to_string()));
            }
            let byte = data[pos];
            pos += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
        }
        result.push(value);
    }
    Ok(result)
}


pub fn matchers_to_expr(
    label_matchers: Matchers,
) -> Result<Vec<Expr>> {
    let mut exprs = Vec::with_capacity(label_matchers.matchers.len());
    for matcher in label_matchers.matchers {

        let col = Expr::Column(Column::from_name(matcher.name));
        let lit = Expr::Literal(ScalarValue::Utf8(Some(matcher.value)), None);
        let expr = match matcher.op {
            MatchOp::Equal => col.eq(lit),
            MatchOp::NotEqual => col.not_eq(lit),
            MatchOp::Re(re) => {

                // This is a hack to handle `.+` and `.*`, and is not strictly correct
                // `.` doesn't match newline (`\n`). Given this is in PromQL context,
                // most of the time it's fine.
                if re.as_str() == "^(?:.*)$" {
                    continue;
                }
                if re.as_str() == "^(?:.+)$" {
                    col.not_eq(Expr::Literal(
                        ScalarValue::Utf8(Some(String::new())),
                        None,
                    ))
                } else {
                    Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(col),
                        op: Operator::RegexMatch,
                        right: Box::new(Expr::Literal(
                            ScalarValue::Utf8(Some(re.as_str().to_string())),
                            None,
                        )),
                    })
                }
            }
            MatchOp::NotRe(re) => {
                if re.as_str() == "^(?:.*)$" {
                    Expr::Literal(ScalarValue::Boolean(Some(false)), None)
                } else if re.as_str() == "^(?:.+)$" {
                    col.eq(Expr::Literal(
                        ScalarValue::Utf8(Some(String::new())),
                        None,
                    ))
                } else {
                    Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(col),
                        op: Operator::RegexNotMatch,
                        right: Box::new(Expr::Literal(
                            ScalarValue::Utf8(Some(re.as_str().to_string())),
                            None,
                        )),
                    })
                }
            }
        };
        exprs.push(expr);
    }

    Ok(exprs)
}

async fn setup_file_reader(file_path: &PathBuf) -> Result<(Box<dyn AsyncFileReader>, ArrowReaderMetadata, ParquetFileMetrics, ExecutionPlanMetricsSet, Arc<dyn ParquetFileReaderFactory>, FileMeta)> {
    // Open the file for reading
    let metadata = fs::metadata(file_path.as_path()).expect("Local file metadata");
    let object_meta = ObjectMeta {
        location: object_store::path::Path::from(file_path.to_str().unwrap()),
        last_modified: metadata.modified().map(|t| t.into()).unwrap(),
        size: metadata.len(),
        e_tag: None,
        version: None,
    };
    let file_meta = FileMeta::from(object_meta);

    let file_name = file_meta.location().as_ref().to_string();

    // Set up local file system object store
    let object_store = Arc::new(LocalFileSystem::new());
    let parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory> = Arc::new(DefaultParquetFileReaderFactory::new(object_store));
    
    let metrics = ExecutionPlanMetricsSet::new();
    let file_metrics = ParquetFileMetrics::new(
        0,
        &file_name,
        &metrics,
    );

    let mut async_file_reader: Box<dyn AsyncFileReader> = parquet_file_reader_factory.create_reader(
        0,
        file_meta.clone(),
         None,
         &metrics,
     )?;

     let options = ArrowReaderOptions::new().with_page_index(true);

    // Begin by loading the metadata from the underlying reader (note
    // the returned metadata may actually include page indexes as some
    // readers may return page indexes even when not requested -- for
    // example when they are cached)
    let reader_metadata = ArrowReaderMetadata::load_async(&mut async_file_reader, options.clone())
        .await?;

    Ok((async_file_reader, reader_metadata, file_metrics, metrics, parquet_file_reader_factory, file_meta))
}

async fn search(
    parquet_file_reader_factory: &Arc<dyn ParquetFileReaderFactory>,
    file_meta: &FileMeta,
    reader_metadata: &ArrowReaderMetadata,
    metrics: &ExecutionPlanMetricsSet,
    file_metrics: ParquetFileMetrics,
    matchers: Matchers,
) -> Result<Option<RowSelection>> {
    let reorder_predicates = true;

        let exprs = matchers_to_expr(matchers)?;
        let expr = conjunction(exprs).unwrap();

        // Create a new async_file_reader using the factory
        let async_file_reader = parquet_file_reader_factory.create_reader(
            0,
            file_meta.clone(),
            None,
            &metrics,
        )?;
        
        // Note: The actual reading would happen here, but we're just demonstrating the setup
        // In a real implementation, you would use the parquet_file_reader_factory to read the file
        let mut builder = SearchReaderBuilder::new_builder(async_file_reader, reader_metadata.clone());


            // Note about schemas: we are actually dealing with **3 different schemas** here:
            // - The table schema as defined by the TableProvider.
            //   This is what the user sees, what they get when they `SELECT * FROM table`, etc.
            // - The logical file schema: this is the table schema minus any hive partition columns and projections.
            //   This is what the physicalfile schema is coerced to.
            // - The physical file schema: this is the schema as defined by the parquet file. This is what the parquet file actually contains.
            let physical_file_schema = Arc::clone(builder.schema());

            let predicate_creation_errors = MetricBuilder::new(&metrics)
            .global_counter("num_predicate_creation_errors");

            let predicate: Arc<dyn PhysicalExpr> = logical2physical(&expr, &builder.schema());

                    // Build predicates for this specific file
                    let (pruning_predicate, page_pruning_predicate) = build_pruning_predicates(
                        Some(&predicate),
                        &physical_file_schema,
                        &predicate_creation_errors,
                    );

                    let schema_adapter_factory: Arc<dyn SchemaAdapterFactory> = Arc::new(DefaultSchemaAdapterFactory);

                 // Filter pushdown: evaluate predicates during scan
                    let row_filter = build_row_filter(
                        &predicate,
                        &physical_file_schema,
                        &physical_file_schema,
                        builder.metadata(),
                        reorder_predicates,
                        &file_metrics,
                        &schema_adapter_factory,
                    );
    
                    match row_filter {
                        Ok(Some(filter)) => {
                            builder = builder.with_row_filter(filter);
                        }
                        Ok(None) => {}
                        Err(e) => {
                            debug!(
                                "Ignoring error building row filter for '{predicate:?}': {e}"
                            );
                        }
                    };       

                // Determine which row groups to actually read. The idea is to skip
            // as many row groups as possible based on the metadata and query
            let file_metadata = Arc::clone(builder.metadata());
            let predicate = pruning_predicate.as_ref().map(|p| p.as_ref());
            let rg_metadata = file_metadata.row_groups();
            // track which row groups to actually read
            let access_plan = ParquetAccessPlan::new_all(rg_metadata.len());
                let mut row_groups = RowGroupAccessPlanFilter::new(access_plan);
    
            // If there is a predicate that can be evaluated against the metadata
            if let Some(predicate) = predicate.as_ref() {
                    row_groups.prune_by_statistics(
                        &physical_file_schema,
                        builder.parquet_schema(),
                        rg_metadata,
                        predicate,
                        &file_metrics,
                    );
            }

                let mut access_plan = row_groups.build();


                // page index pruning: if all data on individual pages can
                // be ruled using page metadata, rows from other columns
                // with that range can be skipped as well
                if !access_plan.is_empty() {
                    if let Some(p) = page_pruning_predicate {
                        access_plan = p.prune_plan_with_page_index(
                            access_plan,
                            &physical_file_schema,
                            builder.parquet_schema(),
                            file_metadata.as_ref(),
                            &file_metrics,
                        );
                    }
                }
    
                let row_group_indexes = access_plan.row_group_indexes();
                if let Some(row_selection) =
                    access_plan.into_overall_row_selection(rg_metadata)?
                {
                    builder = builder.with_row_selection(row_selection);
                }
                
                // Set row groups to read based on the access plan
                if !row_group_indexes.is_empty() {
                    builder = builder.with_row_groups(row_group_indexes);
                }
                
                let batch_size = 1000;

                // Build the SearchReader
                let search_reader = builder.build()?;
                
                // Read the first row group if available
                if let Some(first_row_group_idx) = search_reader.row_groups().front().copied() {
                    println!("Reading row group {} with batch size {}", first_row_group_idx, batch_size);
                    
                    // Clone the selection before moving search_reader
                    let selection = search_reader.selection.clone();
                    
                    // Call read_row_group to process the first row group
                    let result = search_reader.read_row_group(
                        first_row_group_idx,
                        selection,
                        batch_size,
                    ).await?;
                    
                    match result {
                        Some(row_selection) => {
                            println!("Row group {} processed successfully. Row selection: {} rows", 
                                first_row_group_idx, row_selection.row_count());
                            return Ok(Some(row_selection));
                        }
                        None => {
                            println!("Row group {} was filtered out (no rows selected)", first_row_group_idx);
                            return Ok(None);
                        }
                    }
                } else {
                    println!("No row groups to read");
                    return Ok(None);
                }
}

async fn materialize(
    input: Box<dyn AsyncFileReader>,
    parquet_file_reader_factory: &Arc<dyn ParquetFileReaderFactory>,
    file_meta: &FileMeta,
    metrics: &ExecutionPlanMetricsSet,
    metadata: ArrowReaderMetadata,
    row_selection: Option<&RowSelection>, 
    projection: Option<&[String]>,
) -> Result<()> {
    // First, read only the column indexes column to get the schema and column indexes
    let builder = ArrowReaderBuilder::new_with_metadata_async_reader(input, metadata.clone());
    let parquet_schema = metadata.parquet_schema();
    
    // Create projection mask for only the column indexes column
    let col_indexes_mask = ProjectionMask::columns(&parquet_schema, ["s_col_indexes"]);

    let col_indexes_reader = {
        let mut reader_builder = builder.with_projection(col_indexes_mask);
        
        // Apply row selection if provided
        if let Some(selection) = row_selection {
            reader_builder = reader_builder.with_row_selection(selection.clone());
        }
        
        reader_builder.build()?
    };
    
    println!("Reading column indexes first...");
    let mut all_column_indexes = Vec::new();
    let mut total_rows = 0;
    
    // Read only the column indexes column
    let mut stream = col_indexes_reader;
    while let Some(maybe_batch) = stream.next().await {
        let batch = maybe_batch?;
        total_rows += batch.num_rows();
        
        if let Some(col_indexes_col) = batch.column_by_name("s_col_indexes") {
            if let Some(binary_array) = col_indexes_col.as_any().downcast_ref::<BinaryArray>() {
                for row_idx in 0..batch.num_rows() {
                    if binary_array.is_valid(row_idx) {
                        let binary_data = binary_array.value(row_idx);
                        match decode_int_slice(binary_data) {
                            Ok(indexes) => {
                                all_column_indexes.push(indexes);
                            }
                            Err(e) => {
                                println!("Error decoding column indexes for row {}: {}", total_rows - batch.num_rows() + row_idx, e);
                                all_column_indexes.push(Vec::new());
                            }
                        }
                    } else {
                        all_column_indexes.push(Vec::new());
                    }
                }
            }
        }
    }
    
    println!("Total rows: {}", total_rows);
    println!("Column indexes decoded: {}", all_column_indexes.len());
    
    // Build column mappings: which columns are needed for which row selections
    let selection = if let Some(selection) = row_selection {
        selection.clone()
    } else {
        RowSelection::from(vec![
            RowSelector::select(total_rows)
        ])
    };
    let column_to_ranges = build_column_mappings(selection, &all_column_indexes);
    
    println!("Column mappings built:");
    for (col_idx, ranges) in &column_to_ranges {
        let row_count: usize = ranges.iter().map(|r| r.end - r.start).sum();
        println!("  Column {}: needed for {} rows", col_idx, row_count);
        println!("    Ranges: {:?}", ranges);
    }
    
    // Get the total number of rows in the Parquet file
    let full_schema = metadata.schema();
    let total_file_rows: usize = metadata.metadata().file_metadata().num_rows() as usize;
    
    println!("Total file rows: {}", total_file_rows);
    
    // Determine which columns to materialize
    let columns_to_materialize = if let Some(projection) = projection {
        // Use specified projection
        println!("Using specified projection: {:?}", projection);
        projection.to_vec()
    } else {
        // Use all columns that have data (from column indexes)
        println!("Using all columns with data (from column indexes)");
        column_to_ranges
            .keys()
            .filter_map(|&col_idx| {
                if col_idx < full_schema.fields().len() {
                    Some(full_schema.fields()[col_idx].name().clone())
                } else {
                    None
                }
            })
            .collect()
    };
    
    println!("Columns to materialize: {:?}", columns_to_materialize);
    
    // Collect all column data first, then print each series (row) together
    let mut row_data: HashMap<usize, HashMap<String, String>> = HashMap::new();

    // Prepare column reading tasks for concurrent execution
    let mut column_tasks = Vec::new();
    
    for (col_idx, ranges) in &column_to_ranges {
        if *col_idx >= full_schema.fields().len() {
            println!("Skipping invalid column index: {}", col_idx);
            continue;
        }
        
        let field = &full_schema.fields()[*col_idx];
        let col_name = field.name();
        
        // Check if this column should be materialized
        if !columns_to_materialize.contains(col_name) {
            println!("Skipping column '{}' (not in projection)", col_name);
            continue;
        }
        
        let row_count: usize = ranges.iter().map(|r| r.end - r.start).sum();
        println!("Reading column '{}' (index {}) for {} selected rows", col_name, col_idx, row_count);
        
        // Clone necessary data for the async task
        let ranges_clone = ranges.clone();
        let file_meta_clone = file_meta.clone();
        let metadata_clone = metadata.clone();
        let all_column_indexes_clone = all_column_indexes.clone();
        let parquet_file_reader_factory_clone = parquet_file_reader_factory.clone();
        
        // Create async task for reading this column
        let task = read_column_data(
            *col_idx,
            col_name.clone(),
            ranges_clone,
            total_file_rows,
            &parquet_schema,
            file_meta_clone,
            metrics,
            metadata_clone,
            all_column_indexes_clone,
            parquet_file_reader_factory_clone,
        );
        
        column_tasks.push(task);
    }
    
    // Execute all column reading tasks concurrently
    println!("Starting concurrent reading of {} columns...", column_tasks.len());
    let column_results = join_all(column_tasks).await;
    
    // Collect results from all concurrent column reads
    for result in column_results {
        match result {
            Ok(column_data) => {
                // Merge the column data into the main row_data HashMap
                for (row_idx, row_columns) in column_data {
                    for (col_name, value) in row_columns {
                        row_data.entry(row_idx)
                            .or_insert_with(HashMap::new)
                            .insert(col_name, value);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading column: {}", e);
                return Err(e);
            }
        }
    }
    
    // Print each series (row) together
    println!("\n=== Materialized Series ===");
    let mut sorted_rows: Vec<_> = row_data.iter().collect();
    sorted_rows.sort_by_key(|(row_idx, _)| **row_idx);
    
    for (row_idx, columns) in sorted_rows {
        print!("Row {}: ", row_idx);
        let mut column_pairs: Vec<_> = columns.iter().collect();
        column_pairs.sort_by_key(|(col_name, _)| *col_name);
        
        for (i, (col_name, value)) in column_pairs.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}={}", col_name, value);
        }
        println!();
    }
    
    Ok(())
}

// read_column_data reads a single column with its associated row ranges
async fn read_column_data(
    col_idx: usize,
    col_name: String,
    ranges: Vec<std::ops::Range<usize>>,
    total_file_rows: usize,
    parquet_schema: &SchemaDescriptor,
    file_meta: FileMeta,
    metrics: &ExecutionPlanMetricsSet,
    metadata: ArrowReaderMetadata,
    all_column_indexes: Vec<Vec<u64>>,
    parquet_file_reader_factory: Arc<dyn ParquetFileReaderFactory>,
) -> Result<HashMap<usize, HashMap<String, String>>> {
    let mut column_row_data: HashMap<usize, HashMap<String, String>> = HashMap::new();
    
    // Create projection mask for this specific column
    let col_mask = ProjectionMask::columns(&parquet_schema, [col_name.as_str()]);
    
    // Convert Vec<Range> to RowSelection for the reader
    let row_selection = RowSelection::from_consecutive_ranges(ranges.iter().cloned(), total_file_rows);
    debug!("Row selection: {:?}", row_selection);
    
    let async_file_reader: Box<dyn AsyncFileReader> = parquet_file_reader_factory.create_reader(
        0,
        file_meta,
        None,
        &metrics,
    )?;
    
    // Create a new builder for this column
    let col_builder = ArrowReaderBuilder::new_with_metadata_async_reader(async_file_reader, metadata);
    
    // Apply row selection to only read the rows we need
    let col_reader = col_builder
        .with_projection(col_mask)
        .with_row_selection(row_selection)
        .build()?;
    
    // Read the column data
    let mut current_row = 0;
    let mut stream = col_reader;
    let mut batch_idx = 0;
    
    while let Some(maybe_batch) = stream.next().await {
        let batch = maybe_batch?;
        println!("  Column '{}' Batch {}: {} rows", col_name, batch_idx, batch.num_rows());
        
        // Process rows in this batch
        for row_idx in 0..batch.num_rows() {
            let global_row_idx = current_row + row_idx;
            
            if global_row_idx < all_column_indexes.len() {
                let column_indexes = &all_column_indexes[global_row_idx];
                
                // Check if this row actually needs this column
                if column_indexes.contains(&(col_idx as u64)) {
                    if let Some(array) = batch.column_by_name(&col_name) {
                        if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                            let value = if string_array.is_valid(row_idx) {
                                string_array.value(row_idx).to_string()
                            } else {
                                "<null>".to_string()
                            };
                            
                            // Store the value for this row and column
                            column_row_data.entry(global_row_idx)
                                .or_insert_with(HashMap::new)
                                .insert(col_name.clone(), value);
                        }
                    }
                }
            }
        }
        current_row += batch.num_rows();
        batch_idx += 1;
    }
    
    Ok(column_row_data)
}

// build_column_mappings creates mappings between columns and row ranges based on column indexes
fn build_column_mappings(rs: RowSelection, column_indexes: &[Vec<u64>]) -> HashMap<usize, Vec<std::ops::Range<usize>>> {
    let mut column_to_ranges: HashMap<usize, Vec<std::ops::Range<usize>>> = HashMap::new();
    
    let mut column_index_pos = 0;
    let mut current_row = 0;
    
    for selector in rs.iter() {
        if !selector.skip {
            // This is a select operation - process rows in this range
            let range_start = current_row;
            let range_count = selector.row_count as usize;
            let range_end = range_start + range_count;
            
            // Track which columns we've seen in this range to avoid duplicates
            let mut seen_columns = std::collections::HashSet::new();
            
            // Process each row in the current range
            for _row_in_range in range_start..range_end {
                if column_index_pos >= column_indexes.len() {
                    break; // Safety check
                }
                
                let column_ids = &column_indexes[column_index_pos];
                column_index_pos += 1;
                
                // Track which columns are needed for this row range
                for &column_id in column_ids {
                    let col_idx = column_id as usize;
                    
                    if !seen_columns.contains(&col_idx) {
                        // Add a range for this specific column
                        let range = range_start..range_end;
                        
                        // Add range to the vector for this column
                        column_to_ranges
                            .entry(col_idx)
                            .or_insert_with(Vec::new)
                            .push(range);
                        
                        seen_columns.insert(col_idx);
                    }
                }
            }
        }
        current_row += selector.row_count as usize;
    }
    
    column_to_ranges
}


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Parse projection parameter if provided
    let projection = if let Some(projection_str) = &args.projection {
        Some(parse_projection_string(projection_str)?)
    } else {
        None
    };

    println!("Processing file: {}", args.path.display());
    
    let (async_file_reader, reader_metadata, file_metrics, metrics, parquet_file_reader_factory, file_meta) = setup_file_reader(&args.path).await?;
    
    // Check if matcher is provided
    if let Some(matcher_str) = &args.matcher {
        println!("Using matcher: {}", matcher_str);
        
        // Parse matcher string into Matchers
        let matchers = parse_matcher_string(matcher_str)?;
        
        let row_selection = search(&parquet_file_reader_factory, &file_meta, &reader_metadata, &metrics, file_metrics, matchers).await?;
        
        match row_selection {
            Some(selection) => {
                println!("Search found {} matching rows", selection.row_count());
                materialize(async_file_reader, &parquet_file_reader_factory, &file_meta, &metrics, reader_metadata, Some(&selection), projection.as_deref()).await?;
            }
            None => {
                println!("No matching rows found");
            }
        }
    } else {
        println!("No matcher provided, materializing all rows");
        materialize(async_file_reader, &parquet_file_reader_factory, &file_meta, &metrics, reader_metadata, None, projection.as_deref()).await?;
    }
    
    Ok(())
    
}

pub fn build_pruning_predicates(
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    file_schema: &SchemaRef,
    predicate_creation_errors: &Count,
) -> (
    Option<Arc<PruningPredicate>>,
    Option<Arc<PagePruningAccessPlanFilter>>,
) {
    let Some(predicate) = predicate.as_ref() else {
        return (None, None);
    };
    let pruning_predicate = build_pruning_predicate(
        Arc::clone(predicate),
        file_schema,
        predicate_creation_errors,
    );
    let page_pruning_predicate = build_page_pruning_predicate(predicate, file_schema);
    (pruning_predicate, Some(page_pruning_predicate))
}

/// Build a page pruning predicate from an optional predicate expression.
/// If the predicate is None or the predicate cannot be converted to a page pruning
/// predicate, return None.
pub fn build_page_pruning_predicate(
    predicate: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Arc<PagePruningAccessPlanFilter> {
    Arc::new(PagePruningAccessPlanFilter::new(
        predicate,
        Arc::clone(file_schema),
    ))
}
