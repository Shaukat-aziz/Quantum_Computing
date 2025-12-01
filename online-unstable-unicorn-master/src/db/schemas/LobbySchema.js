const mongoose = require("mongoose");
const { PlayerSchema } = require("./PlayerSchema");
const { CardSchema } = require("./CardSchema");
const { GameSchema } = require("./GameSchema");

const LobbySchema = new mongoose.Schema({
  key: String,
  name: String,
  uri: String,
  players: Number,
  currentPlayers: [PlayerSchema],
  currentGame: GameSchema,
  currentDecks: [PlayerSchema],
});

const Lobby = mongoose.model("lobbies", LobbySchema);

exports.GameSchema = LobbySchema;
exports.Lobby = Lobby;
