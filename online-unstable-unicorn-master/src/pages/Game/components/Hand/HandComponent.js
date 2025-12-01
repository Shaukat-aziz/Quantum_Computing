import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  playingCard,
  attemptToPlay,
  discardCard,
  discardingCard,
} from "actions";
import { useMyPlayer } from "utils/hooks.js";
import groupBy from "lodash/groupBy";
import { Card, Header } from "semantic-ui-react";
import "./HandComponent.css";

import CardComponent from "components/Card/CardComponent";

function HandComponent() {
  const myPlayer = useMyPlayer();
  const socketServer = useSelector((state) => state.socket);
  const isMyTurn = useSelector((state) => state.isMyTurn);
  const isPlayingCard = useSelector((state) => state.isPlayingCard);
  const isDiscardingCard = useSelector((state) => state.isDiscardingCard);
  const players = useSelector((state) => state.players);
  const decks = useSelector((state) => state.decks);
  const lobbyName = useSelector((state) => state.game.uri);
  const dispatch = useDispatch();

  useEffect(() => {
    if (isPlayingCard.basicUnicornOnly) {
      if (!groupBy(myPlayer.hand, "type")["Basic Unicorn"]) {
        isPlayingCard.callback();
      }
    }
  }, [isPlayingCard, isPlayingCard.basicUnicornOnly, myPlayer.hand]);

  function handleOnClick(card, index) {
    if (isPlayingCard.isTrue) {
      handlePlayCard(card, index);
    }

    if (isDiscardingCard.isTrue) {
      handleDiscardCard(card, index);
    }
  }

  function handlePlayCard(card, index) {
    if (isPlayingCard.basicUnicornOnly) {
      if (card.type === "Basic Unicorn") {
        dispatch(playingCard(false));
        dispatch(attemptToPlay(card));
        socketServer.emit("attemptToPlayCard", lobbyName, card);
      }
    } else {
      switch (card.type) {
        case "Baby Unicorn":
        case "Basic Unicorn":
        case "Magical Unicorn":
        case "Upgrade":
        case "Downgrade":
        case "Magic":
          dispatch(playingCard(false));
          dispatch(attemptToPlay(card));
          socketServer.emit("attemptToPlayCard", lobbyName, card);
          break;
        default:
          console.log("No Card found");
      }
    }
  }

  function handleDiscardCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    updatedDecks["discardPile"].push(card);
    updatedPlayers[myPlayer.currentPlayerIndex].hand.splice(index, 1);
    socketServer.emit(
      "discardCard",
      lobbyName,
      card,
      updatedDecks,
      updatedPlayers
    );
    if (isDiscardingCard.callback) {
      isDiscardingCard.callback();
    }

    dispatch(discardingCard({ isTrue: false, callback: null }));
  }

  return (
    <div className="hand">
      {isPlayingCard.isTrue ? <Header>Choose Card to Play</Header> : null}
      {isDiscardingCard.isTrue ? <Header>Choose Card to Discard</Header> : null}
      <Card.Group>
        {myPlayer.hand.map((card, index) => {
          return (
            <CardComponent
              index={index}
              key={card.id}
              card={card}
              basicUnicornOnly={isPlayingCard.basicUnicornOnly}
              callback={handleOnClick}
            />
          );
        })}
      </Card.Group>
    </div>
  );
}

export default HandComponent;
