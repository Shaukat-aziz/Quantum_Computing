import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import "./LobbyPage.scss";
import { setPlayers, startGame, joinLobby, leaveLobby } from "actions";
import { useHistory, useParams } from "react-router-dom";
import { useMyPlayer } from "utils/hooks.js";
import GroupBy from "lodash/groupBy";
import Remove from "lodash/remove";
import Shuffle from "lodash/shuffle";
import { Segment, Button, Label, Message } from "semantic-ui-react";

// Components
import AddPlayer from "./components/AddPlayer/AddPlayer";
import PlayersView from "components/PlayersView/PlayersView";

function LobbyPage(props) {
  const myPlayer = useMyPlayer();
  const socketServer = useSelector((state) => state.socket);
  const game = useSelector((state) => state.game);
  const players = useSelector((state) => state.players);
  const [babyUnicorns, setBabyUnicorns] = useState(
    GroupBy(game.cards, "type")["Baby Unicorn"]
  );
  const [inLobby, setLobby] = useState(0);
  const [showMessage, setShowMessage] = useState(false);
  const history = useHistory();
  const urlParams = useParams().id;
  const dispatch = useDispatch();

  useEffect(() => {
    if (!game.roomId) {
      socketServer.on("reconnect", (lobby, gameState) => {
        if (gameState) {
          dispatch(joinLobby(lobby));
          setLobby(inLobby);
          dispatch(setPlayers(gameState.currentPlayers));
        } else {
          history.push("/app");
        }

        socketServer.removeListener("reconnect");
      });
      socketServer.emit("checkForRoom", urlParams);
    }

    socketServer.on("userConnected", (inLobby, inGame) => {
      setLobby(inLobby);
      dispatch(setPlayers(inGame));

      console.log(inGame);
    });

    socketServer.on("playerAdded", (players) => {
      dispatch(setPlayers(players));
    });

    socketServer.on("startingGame", (options, decks, players) => {
      dispatch(
        startGame(
          options,
          decks,
          players,
          localStorage.getItem("currentPlayerIndex") === "0"
        )
      );
      history.push(`/app/${urlParams}/game`);
    });

    return () => {
      socketServer.removeListener("userConnected");
      socketServer.removeListener("reconnect");
      socketServer.removeListener("playerAdded");
      socketServer.removeListener("startingGame");
    };
  }, [dispatch, game.roomId, history, inLobby, socketServer, urlParams]);

  useEffect(() => {
    if (inLobby === players.length) {
      setShowMessage(false);
    }
  }, [inLobby, players.length]);

  function renderMessage() {
    if (showMessage && myPlayer.currentPlayerIndex === "0") {
      return (
        <Message floating color="blue">
          {inLobby - players.length} player(s) left to make characters
        </Message>
      );
    }
  }

  function renderStartGameButton() {
    if (myPlayer.currentPlayerIndex === "0") {
      return (
        <Button
          id="start-game-button"
          onClick={handleStartGame}
          color="blue"
          size="massive"
        >
          Start Game
        </Button>
      );
    }
  }

  function renderWaitingMessage() {
    if (myPlayer.currentPlayerIndex !== "0") {
      return (
        <Message floating color="blue">
          Waiting for game leader to start game ....
        </Message>
      );
    }
  }

  function handleLeaveLobby() {
    socketServer.emit("leaveLobby", urlParams);
    dispatch(leaveLobby());
    history.push(`/app`);
  }

  function deal(deck, currentPlayers = players) {
    let dealAmount = currentPlayers.length < 6 ? 5 : 6;
    for (let i = 0; i < currentPlayers.length * dealAmount; i++) {
      const playerIndex = i % currentPlayers.length;
      const card = deck.splice(0, 1)[0];
      currentPlayers[playerIndex].hand.push(card);
    }
    return [deck, players];
  }

  function handleStartGame() {
    let cards = game.cards;
    let currentPlayers = players;

    Remove(cards, (c) => {
      return c.type === "Baby Unicorn";
    });

    const [drawPile, updatedPlayers] = deal(Shuffle(cards), currentPlayers);
    socketServer.emit(
      "startGame",
      urlParams,
      {
        ...game,
        whosTurn: currentPlayers[0],
        playing: true,
      },
      {
        drawPile,
        nursery: babyUnicorns,
        discardPile: [],
      },
      updatedPlayers
    );
  }

  return (
    <div>
      <Segment id="lobby-view">
        <Label
          attached="top right"
          content="Leave Lobby"
          onClick={handleLeaveLobby}
          color="red"
        />

        {renderMessage()}
        <AddPlayer />
        {renderStartGameButton()}
        {renderWaitingMessage()}
      </Segment>

      <PlayersView />
    </div>
  );
}

export default LobbyPage;
