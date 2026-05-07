//! Arrow-native schemas for the Unified Relational Semantic Abstraction.
//!
//! Raw unstructured data (images, video, text) is automatically projected into
//! strongly-typed Arrow schemas so the symbolic logic layer can reason over it.

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

// ── Scene-Graph schemas (Image / Video) ────────────────────────────────────

/// Schema for detected objects in a visual scene.
///
/// Each row is one bounding-box detection with a class label and confidence.
pub fn objects_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("frame_id", DataType::UInt64, true),
        Field::new("class_label", DataType::Utf8, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("bbox_x_min", DataType::Float32, false),
        Field::new("bbox_y_min", DataType::Float32, false),
        Field::new("bbox_x_max", DataType::Float32, false),
        Field::new("bbox_y_max", DataType::Float32, false),
        Field::new("source_model_id", DataType::Utf8, false),
        Field::new("source_model_ver", DataType::Utf8, false),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

/// Schema for spatial / temporal relationships between detected objects.
pub fn scene_relationships_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("rel_id", DataType::Utf8, false),
        Field::new("subject_id", DataType::Utf8, false),
        Field::new("object_id", DataType::Utf8, false),
        Field::new("predicate", DataType::Utf8, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("frame_id", DataType::UInt64, true),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

/// Schema for per-object attributes (colour, material, pose, etc.).
pub fn attributes_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("attribute_key", DataType::Utf8, false),
        Field::new("attribute_value", DataType::Utf8, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

// ── Entity-Graph schemas (Text) ────────────────────────────────────────────

/// Schema for resolved entities across documents.
pub fn entities_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Utf8, false),
        Field::new("entity_type", DataType::Utf8, false),
        Field::new("canonical_name", DataType::Utf8, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("source_model_id", DataType::Utf8, false),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

/// Schema for character-span mentions linking back to source documents.
pub fn mentions_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("mention_id", DataType::Utf8, false),
        Field::new("entity_id", DataType::Utf8, false),
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("span_start", DataType::UInt64, false),
        Field::new("span_end", DataType::UInt64, false),
        Field::new("surface_form", DataType::Utf8, false),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

/// Schema for entity-to-entity relationships extracted from text.
pub fn text_relationships_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("rel_id", DataType::Utf8, false),
        Field::new("subject_entity_id", DataType::Utf8, false),
        Field::new("object_entity_id", DataType::Utf8, false),
        Field::new("predicate", DataType::Utf8, false),
        Field::new("confidence", DataType::Float64, false),
        Field::new("doc_id", DataType::Utf8, true),
        Field::new("provenance", DataType::Binary, true),
    ]))
}

// ── AI-Tables system catalog ───────────────────────────────────────────────

/// Schema for the `__ai_models` system catalog table.
pub fn ai_models_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("model_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("version", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("input_schema_json", DataType::Utf8, true),
        Field::new("output_schema_json", DataType::Utf8, true),
        Field::new("avg_latency_ms", DataType::Float64, false),
        Field::new("accuracy", DataType::Float64, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("device_affinity", DataType::Utf8, true),
        Field::new("created_at", DataType::Utf8, false),
        Field::new("updated_at", DataType::Utf8, false),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_have_provenance_column() {
        for (name, schema) in [
            ("objects", objects_schema()),
            ("scene_rels", scene_relationships_schema()),
            ("attrs", attributes_schema()),
            ("entities", entities_schema()),
            ("mentions", mentions_schema()),
            ("text_rels", text_relationships_schema()),
        ] {
            assert!(
                schema.column_with_name("provenance").is_some(),
                "{name} schema missing provenance column"
            );
        }
    }

    #[test]
    fn ai_models_schema_has_required_fields() {
        let schema = ai_models_schema();
        for field in [
            "model_id",
            "name",
            "version",
            "format",
            "accuracy",
            "avg_latency_ms",
        ] {
            assert!(
                schema.column_with_name(field).is_some(),
                "ai_models schema missing {field}"
            );
        }
    }
}
