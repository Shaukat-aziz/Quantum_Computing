const mongoose = require("mongoose");
const { EffectSchema } = require("./EffectSchema");

const CardSchema = new mongoose.Schema({
  _id: Number,
  name: String,
  type: String,
  description: String,
  color: String,
  url: String,
  activateAtBeginning: Boolean,
  upgrade: EffectSchema,
  downgrade: EffectSchema,
  activateAtEnd: Boolean,
  activateOnPlay: Boolean,
  playRequirement: String,
  isGlobal: Boolean,
});

const Card = mongoose.model("cards", CardSchema);

exports.CardSchema = CardSchema;
exports.Card = Card;
