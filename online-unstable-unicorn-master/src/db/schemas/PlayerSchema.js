const mongoose = require("mongoose");
const { CardSchema } = require("./CardSchema");
const { EffectSchema } = require("./EffectSchema");

const PlayerSchema = new mongoose.Schema({
  index: Number,
  connected: Boolean,
  color: String,
  name: String,
  created: { type: Date, default: Date.now },
  hand: [CardSchema],
  stable: [CardSchema],
  viewingStableId: Number,
  viewingOtherPlayerModalId: Number,
  unicorn: CardSchema,
  upgrades: [EffectSchema],
  downgrades: [EffectSchema],
});

const Player = mongoose.model("players", PlayerSchema);

exports.PlayerSchema = PlayerSchema;
exports.Player = Player;
