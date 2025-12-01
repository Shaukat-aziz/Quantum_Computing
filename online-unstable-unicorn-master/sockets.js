const mongo = require("./src/utils/mongo");
const { Lobby } = require("./src/db/schemas/LobbySchema");

module.exports = (server) => {
  const io = require("socket.io").listen(server);

  let connectedUsers = 0;

  let lobbies = [];

  // TODO: Make API call for this
  let games = {
    unicornsarelit: {
      currentGame: {},
      currentDecks: {},
      currentPlayers: [],
    },
  };

  function getLobbies(lobbies) {
    return Object.keys(lobbies).reduce((currentLobbies, key) => {
      if (key.match(/game/gi)) {
        return [
          ...currentLobbies,
          {
            key: key,
            name: key.split(":")[1],
            uri: removeSpaces(key.split(":")[1]),
            players: lobbies[key].length,
          },
        ];
      }

      return currentLobbies;
    }, []);
  }

  function returnLobbies(updated) {
    lobbies = getLobbies(socket.adapter.rooms);
    socket.broadcast.emit("returnLobbies", getLobbies(socket.adapter.rooms));
  }

  function removeSpaces(string) {
    return string.split(" ").join("");
  }

  function encodeUriToRoomId(uri) {
    return `game:${uri}`;
  }

  function findLobby(uri) {
    const roomId = encodeUriToRoomId(uri);
    return roomId;
  }

  io.on("connection", async (socket) => {
    // Check if any lobbies saved
    await mongo(null, async () => {
      const myLobbies = await Lobby.find({}, function (err, aLobby) {
        if (err) return console.error(err);

        return aLobby;
      });

      games = myLobbies.reduce((result, currentLobby) => {
        result[currentLobby.uri] = currentLobby;
        return result;
      }, {});

      console.log(games);
      myLobbies.forEach((lobby) => {
        socket.join([lobby.key]);
      });

      // console.log(myLobbies);
    });

    lobbies = getLobbies(socket.adapter.rooms);
    socket.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    // if (!currentGame.id) {
    //   console.log(`a player connected, ${++connectedUsers} in lobby`);
    //
    //   socket.broadcast.emit('returnLobbies', getLobbies(socket.adapter.rooms));
    //   console.log(getLobbies(socket.adapter.rooms))
    // } else {
    //   console.log('new spectator');
    // }

    // HOMEPAGE EVENTS
    socket.on("getLobbies", () => {
      socket.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    });

    socket.on("joinLobby", (lobbyName) => {
      const room = `game:${lobbyName}`;
      socket.join([room]);
      lobbies = getLobbies(socket.adapter.rooms);
      currentLobby = lobbies.find((l) => l.key === room);
      io.to(room).emit(
        "userConnected",
        currentLobby.players,
        games[lobbyName].currentPlayers.filter(
          (testPlayer, index) => index < currentLobby.players
        )
      ); // switch back to active users when ready
      socket.broadcast.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    });

    socket.on("createLobby", (lobbyName) => {
      socket.join([`game:${lobbyName}`]);
      games[removeSpaces(lobbyName)] = {
        currentGame: {},
        currentDecks: {},
        currentPlayers: [],
      };
      socket.broadcast.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    });

    socket.on("checkForRoom", (lobbyName) => {
      const room = `game:${lobbyName}`;
      console.log(lobbyName);

      if (games[lobbyName]) {
        console.log("found lobby");
        socket.join([room]);
        lobbies = getLobbies(socket.adapter.rooms);
        currentLobby = lobbies.find((l) => l.key === room);

        if (games[lobbyName].currentGame.playing) {
          console.log("reconnect to game");
          io.to(room).emit(
            "reconnect",
            games[lobbyName].currentGame,
            games[lobbyName].currentDecks,
            games[lobbyName].currentPlayers
          );
        } else {
          games[lobbyName].currentPlayers.filter(
            (testPlayer, index) => index < currentLobby.players
          );
          console.log(
            `Players: ${games[lobbyName].currentPlayers.length}, connected: ${currentLobby.players}`
          );
          socket.emit("reconnect", currentLobby.players, games[lobbyName]);
        }
      } else {
        console.log("no lobby found");
        socket.emit("reconnect", null, null);
      }
    });

    // LOBBY PAGE EVENTS
    socket.on("leaveLobby", (lobbyName) => {
      const room = `game:${lobbyName}`;
      socket.leave([room]);
      lobbies = getLobbies(socket.adapter.rooms);
      currentLobby = lobbies.find((l) => l.key === room);
      if (currentLobby) {
        io.to(room).emit(
          "userConnected",
          currentLobby.players,
          games[lobbyName].currentPlayers.filter(
            (testPlayer, index) => index < currentLobby.players
          )
        );
      } else {
        delete games[lobbyName];
      }
      socket.broadcast.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    });

    socket.on("addPlayer", (lobbyName, newPlayer) => {
      console.log("adding player");
      const room = `game:${lobbyName}`;
      games[lobbyName].currentPlayers.push(newPlayer);
      console.log(
        `players in ${lobbyName}: ${games[lobbyName].currentPlayers.length}`
      );
      io.to(room).emit("playerAdded", games[lobbyName].currentPlayers);
    });

    socket.on("startGame", (lobbyName, game, decks, players) => {
      console.log("Starting Game");
      const room = `game:${lobbyName}`;
      console.log(games[lobbyName]);
      games[lobbyName].currentGame = game;
      games[lobbyName].currentDecks = decks;
      games[lobbyName].currentPlayers = players;
      io.to(room).emit(
        "startingGame",
        games[lobbyName].currentGame,
        games[lobbyName].currentDecks,
        games[lobbyName].currentPlayers
      );

      console.log(games[lobbyName]);
    });

    // GAME EVENTS
    // socket.on("endEffectPhase", (lobbyName, phase) => {
    //   if (games[lobbyName].currentGame.phase === 0) {
    //     console.log("Switching to ", phase, " phase");
    //     games[lobbyName].currentGame.phase = phase;
    //     io.to(`game:${lobbyName}`).emit("switchingPhase", phase);
    //   }
    // });

    // socket.on("drawCard", (lobbyName, decks, players, phase) => {
    //   console.log("drawing card");
    //   games[lobbyName].currentDecks = decks;
    //   games[lobbyName].currentPlayers = players;
    //   games[lobbyName].currentGame.phase = phase;
    //   io.to(`game:${lobbyName}`).emit(
    //     "cardDrew",
    //     games[lobbyName].currentDecks,
    //     games[lobbyName].currentPlayers
    //   );
    //   if (phase) {
    //     io.to(`game:${lobbyName}`).emit("switchingPhase", phase);
    //   }
    // });

    // socket.on("attemptToPlayCard", (lobbyName, card) => {
    //   console.log("Attemping to play: ", card.name);
    //   io.to(`game:${lobbyName}`).emit("attemptCardPlay", card);
    // });

    // socket.on("playCard", (lobbyName, updatedPlayers) => {
    //   console.log("Attemping to play: ", card.name);
    //   io.to(`game:${lobbyName}`).emit("cardPlayed", card, updatedPlayers);
    // });

    // socket.on(
    //   "discardCard",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Discarding Card");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardDiscarded",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "discardingOpponentCard",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Discarding Opponent Card");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardDiscarded",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "destroyCard",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Destroying Card");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardDestroyed",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "sacrificeCard",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Sacrificing Card");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardSacrificed",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on("returnCard", (lobbyName, card, updatedDecks, updatedPlayers) => {
    //   console.log("Returning Card");
    //   io.to(`game:${lobbyName}`).emit(
    //     "cardReturned",
    //     card,
    //     updatedDecks,
    //     updatedPlayers
    //   );
    // });

    // socket.on(
    //   "returningOpponentCard",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Returning Card from stable");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardReturned",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "drawFromOpponent",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("Drawing Card From Opponent");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardDrewFromOpponent",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "giveToOpponent",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("GIVING TO OPPONENT");
    //     io.to(`game:${lobbyName}`).emit(
    //       "cardGivenToPlayer",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on(
    //   "stealUnicorn",
    //   (lobbyName, card, updatedDecks, updatedPlayers) => {
    //     console.log("STEALING UNICORN");
    //     io.to(`game:${lobbyName}`).emit(
    //       "unicornStolen",
    //       card,
    //       updatedDecks,
    //       updatedPlayers
    //     );
    //   }
    // );

    // socket.on("skippingInstant", (lobbyName, playerIndex) => {
    //   console.log("skippingInstant");
    //   io.to(`game:${lobbyName}`).emit("playerCheckedForInstant", playerIndex);
    // });

    // socket.on("playInstant", (lobbyName, playerIndex, instant) => {
    //   console.log("playingIntent: ", instant.name);
    //   io.to(`game:${lobbyName}`).emit(
    //     "playerCheckedForInstant",
    //     playerIndex,
    //     instant
    //   );
    // });

    // socket.on("playersDiscarding", (lobbyName, playerIndexes) => {
    //   console.log("playersDiscarding: ", playerIndexes);
    //   io.to(`game:${lobbyName}`).emit("setPlayersDiscarding", playerIndexes);
    // });

    // socket.on("discardCheck", (lobbyName, playerIndex) => {
    //   console.log("discardCheck: ", playerIndex);
    //   io.to(`game:${lobbyName}`).emit(
    //     "playerCheckedForDiscarding",
    //     playerIndex
    //   );
    // });

    // socket.on("playersSacrificing", (lobbyName, playerIndexes) => {
    //   console.log("playersSacrificing: ", playerIndexes);
    //   io.to(`game:${lobbyName}`).emit("setPlayersSacrificing", playerIndexes);
    // });

    // socket.on("sacrificeCheck", (lobbyName, playerIndex) => {
    //   console.log("sacrificeCheck: ", playerIndex);
    //   io.to(`game:${lobbyName}`).emit("playerCheckedForSacrifing", playerIndex);
    // });

    // socket.on("actionHappened", (lobbyName, updatedDecks, updatedPlayers) => {
    //   console.log("card played");
    //   games[lobbyName].currentPlayers = updatedPlayers;
    //   games[lobbyName].currentDecks = updatedDecks;
    //   io.to(`game:${lobbyName}`).emit(
    //     "updateFromAction",
    //     updatedDecks,
    //     updatedPlayers
    //   );
    // });

    // socket.on("endActionPhase", (lobbyName) => {
    //   console.log("ENDING ACTION PHASE");
    //   games[lobbyName].currentGame.phase = 3;
    //   io.to(`game:${lobbyName}`).emit("endingActionPhase");
    // });

    // socket.on("endTurn", (lobbyName, gameUpdates, nextPlayerIndex) => {
    //   console.log(games[lobbyName].currentGame.phase);
    //   if (games[lobbyName].currentGame.phase === 3) {
    //     console.log("Ending turn");
    //     games[lobbyName].currentGame = {
    //       ...games[lobbyName].currentGame,
    //       ...gameUpdates,
    //     };

    //     io.to(`game:${lobbyName}`).emit(
    //       "endingTurn",
    //       gameUpdates,
    //       nextPlayerIndex
    //     );
    //   }
    // });

    // socket.on("endGame", (msg) => {
    //   console.log("end turn");
    // });

    socket.on("disconnect", () => {
      console.log(`a player left, ${--connectedUsers} in lobby`);
      socket.broadcast.emit("returnLobbies", getLobbies(socket.adapter.rooms));
    });
  });
};
