const mongo = require("./src/utils/mongo");

const { Card } = require("./src/db/schemas/CardSchema.js");
const { Lobby } = require("./src/db/schemas/LobbySchema");
const { Player } = require("./src/db/schemas/PlayerSchema");

// Cards
module.exports = (app) => {
  app.get("/api/cards", function (req, res) {
    mongo(res, async () => {
      return await Card.find(req.query, function (err, dbCards) {
        if (err) return console.error(err);

        return dbCards;
      });
    });
  });

  app.get("/api/lobbies", function (req, res) {
    console.log(req.query);
    mongo(res, async () => {
      return await Lobby.find(req.query, function (err, dbCards) {
        if (err) return console.error(err);

        return dbCards;
      });
    });
  });

  app.get("/api/players", function (req, res) {
    console.log(req.query);
    mongo(res, async () => {
      return await Player.find(req.query, function (err, dbCards) {
        if (err) return console.error(err);

        return dbCards;
      });
    });
  });
};
