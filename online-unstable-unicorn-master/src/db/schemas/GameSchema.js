const mongoose = require("mongoose");
const { PlayerSchema } = require("./PlayerSchema");
const { CardSchema } = require("./CardSchema");
const { EffectSchema } = require("./EffectSchema");

const GameSchema = new mongoose.Schema({
  gameDuration: String,
  expansion: String,
  winCondition: String,
  playing: Boolean,
  gameOver: Boolean,
  phase: Number,
  phases: [
    {
      name: String,
      actions: [String],
    },
  ],
  cards: [CardSchema],
  upgrades: [EffectSchema],
  downgrades: [EffectSchema],
  whosTurn: PlayerSchema,
  turn: Number,
});

const Game = mongoose.model("game", GameSchema);

exports.GameSchema = GameSchema;
exports.Game = Game;
