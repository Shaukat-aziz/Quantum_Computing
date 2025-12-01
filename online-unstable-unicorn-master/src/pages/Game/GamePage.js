import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { startGame } from "actions";
import { useMyPlayer } from "utils/hooks.js";
import { useHistory, useParams } from "react-router-dom";
import "./GamePage.css";

// Components
import PlayersView from "components/PlayersView/PlayersView.js";
import Field from "./components/Field/Field.js";
import StableComponent from "./components/Stable/StableComponent";
import ActionViewComponent from "./components/ActionView/ActionViewComponent.js";
import HandComponent from "./components/Hand/HandComponent";
import ViewOtherPlayer from "./components/ViewOtherPlayer/ViewOtherPlayer.js";

const MemoGamePage = React.memo(GamePage);

function GamePage() {
  const myPlayer = useMyPlayer();
  const isPlaying = useSelector((state) => state.game.playing);
  const socketServer = useSelector((state) => state.socket);
  const lobbyName = useParams().id;
  const history = useHistory();
  const dispatch = useDispatch();

  useEffect(() => {
    if (!isPlaying) {
      socketServer.on("reconnect", (game, decks, players) => {
        if (game.playing) {
          dispatch(
            startGame(
              game,
              decks,
              players,
              game.turn % players.length ===
                parseInt(localStorage.getItem("currentPlayerIndex"))
            )
          );
        } else {
          history.push("/app");
        }

        socketServer.removeListener("reconnect");
      });

      socketServer.emit("checkForRoom", lobbyName);
    }
  }, [dispatch, history, isPlaying, lobbyName, socketServer]);

  // useEffect(() => {
  //   if (gameOver) {
  //     dispatch(endGame(game))
  //     socketServer.emit('endGame', lobbyName, game);
  //     history.push(`/${game.uri}/stats`);
  //   }
  // }, [gameOver]);

  return (
    <div>
      {/* Shows list of players  */}
      <PlayersView />

      {/* Shows decks  */}
      <Field />

      {/* Shows current phase and cards being played  */}
      <ActionViewComponent />

      {/* Allows current player to view another players hand */}
      <ViewOtherPlayer
        isOpen={myPlayer.viewingOtherPlayerModalId != myPlayer.id}
      />

      {/* Shows player's current hand  */}
      <HandComponent />

      {/* Shows player's current stable  */}
      <StableComponent />
    </div>
  );
}

export default MemoGamePage;
