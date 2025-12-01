const mongoose = require("mongoose");
const cards = require("../db/cards.js");
const players = require("../db/data/testplayers");
const lobbies = require("../db/data/testlobby");

const { Card } = require("../db/schemas/CardSchema.js");
const { Player } = require("../db/schemas/PlayerSchema.js");
const { Lobby } = require("../db/schemas/LobbySchema.js");

mongoose.connect("mongodb://localhost/unstableunicorns", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;
db.on("error", console.error.bind(console, "connection error:"));
db.once("open", async function () {
  // await cards.forEach((card) => {
  //   const cardObject = new Card(card);

  //   cardObject.save(function (err, aCard) {
  //     if (err) return console.error(err);
  //   });
  // });

  // await players.forEach(function (player) {
  //   const playerObject = new Player(player);

  //   playerObject.save(function (err, aPlayer) {
  //     if (err) return console.error(err);

  //     return aPlayer;
  //   });
  // });

  await lobbies.forEach(async function (lobby) {
    const lobbyObject = new Lobby(lobby);

    let players = await Player.find({});
    lobbyObject.currentPlayers = players.splice(0, lobby.currentPlayers.length);
    console.log(lobbyObject);

    lobbyObject.save(function (err, aLobby) {
      if (err) return console.error(err);

      return aLobby;
    });
  });

  // db.close();
});

// Test Players
