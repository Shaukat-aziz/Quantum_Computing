const mongoose = require("mongoose");

const EffectSchema = new mongoose.Schema({
  name: String,
  type: String,
  description: String,
  optional: Boolean,
});

const Effect = mongoose.model("effects", EffectSchema);

exports.Effect = Effect;
exports.EffectSchema = EffectSchema;
