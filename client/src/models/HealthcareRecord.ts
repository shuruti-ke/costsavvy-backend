import mongoose, { Schema } from "mongoose";

const HealthcareSchema = new Schema(
  {
    reporting_entity_name_in_network_files: String,
    provider_group_id: Number,
    provider_group_id_type: String,
    sub_npi: Number,
    negotiation_arrangement: String,
    billing_code: Number,
    billing_code_type: String,
    billing_code_name: String,
    billing_code_modifier: String,
    negotiated_type: String,
    negotiated_rate: String,
    billing_class: String,
    provider_name: String,
    provider_city: String,
    provider_state: String,
    provider_zip_code: Number,
    provider_address: String,
    "Description of Service": String,
  },
  {
    collection: "healthcare",
    timestamps: false,
  }
);

export default mongoose.models.HealthcareRecord ||
  mongoose.model("HealthcareRecord", HealthcareSchema);
