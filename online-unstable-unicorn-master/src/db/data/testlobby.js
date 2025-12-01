const players = require("./testplayers");

module.exports = [
  {
    key: "game:unicornsarelit",
    name: "unicorns are lit",
    players: 2,
    uri: "unicornsarelit",
    currentGame: {},
    currentPlayers: [players[0], players[1]],
    currentDecks: [],
  },
  {
    key: "game:threemusketeers",
    name: "three musketeers",
    players: 3,
    uri: "threemusketeers",
    currentGame: {},
    currentPlayers: [players[0], players[1], players[2]],
    currentDecks: [],
  },
  {
    key: "game:fiver",
    name: "fiver",
    players: 4,
    uri: "fiver",
    currentGame: {},
    currentPlayers: players,
    currentDecks: [],
  },
];
