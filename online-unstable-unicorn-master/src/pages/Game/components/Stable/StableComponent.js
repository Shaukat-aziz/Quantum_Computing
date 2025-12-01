import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  returningCard,
  sacrificingCard,
  endGame,
  givingToOpponent,
} from "actions";
import { useMyPlayer } from "utils/hooks.js";
import "./StableComponent.css";
import { Card, Header } from "semantic-ui-react";

// Components
import CardComponent from "components/Card/CardComponent";

// I need to update this comp to check whos stable it should display
const MemoStableComponent = React.memo(() => {
  const currentPlayer = useMyPlayer();
  const { currentPlayerIndex, stable, viewingStableId } = currentPlayer;
  const socketServer = useSelector((state) => state.socket);
  const lobbyName = useSelector((state) => state.game.uri);
  const isReturningCard = useSelector((state) => state.isReturningCard);
  const isSacrificingCard = useSelector((state) => state.isSacrificingCard);
  const isGivingToOpponent = useSelector((state) => state.isGivingToOpponent);
  const [viewStablePlayer, setViewStablePlayer] = useState(currentPlayer);
  const dispatch = useDispatch();

  // Move to player view???
  const players = useSelector((state) => state.players);
  const decks = useSelector((state) => state.decks);

  useEffect(() => {
    const viewingPlayer = players.find(
      (player) => player.id == viewingStableId
    );
    setViewStablePlayer(viewingPlayer);
  }, [players, viewingStableId]);

  useEffect(() => {
    if (stable && stable.length === 7) {
      console.log("ENDING THE GAME");
      dispatch(endGame());
    }
  }, [dispatch, stable]);

  function handleCallback(card, index) {
    if (isReturningCard.isTrue) {
      handleReturnCard(card, index);
    }

    if (isSacrificingCard.isTrue) {
      handleSacrificeCard(card, index);
    }

    if (isGivingToOpponent.isTrue) {
      handleGiveToOpponent(card, index);
    }
  }

  function handleReturnCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    if (card.type === "Baby Unicorn") {
      updatedDecks["nursery"].push(card);
    } else {
      updatedPlayers[currentPlayerIndex].hand.push(card);
    }
    updatedPlayers[currentPlayerIndex].stable.splice(index, 1);

    socketServer.emit(
      "returnCard",
      lobbyName,
      card,
      updatedDecks,
      updatedPlayers
    );
    if (isReturningCard.callback) {
      isReturningCard.callback();
    }
    dispatch(returningCard({ isTrue: false, callback: null }));
  }

  function handleSacrificeCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    if (card.type === "Baby Unicorn") {
      updatedDecks["nursery"].push(card);
    } else {
      updatedDecks["discardPile"].push(card);
    }
    updatedPlayers[currentPlayerIndex].stable.splice(index, 1);

    socketServer.emit(
      "sacrificeCard",
      lobbyName,
      card,
      updatedDecks,
      updatedPlayers
    );
    if (isSacrificingCard.callback) {
      isSacrificingCard.callback();
    }
    dispatch(givingToOpponent({ isTrue: false, callback: null }));
  }

  function handleGiveToOpponent(card, index) {
    if (isGivingToOpponent.callback) {
      isGivingToOpponent.callback({
        card,
        cardIndex: index,
      });
    }
    dispatch(givingToOpponent({ isTrue: false, callback: null }));
  }

  return (
    <div className="stable">
      <Header>{viewStablePlayer ? viewStablePlayer.name : ""}'s Stable</Header>
      {isReturningCard.isTrue ? <Header>Choose Card to Return</Header> : null}
      {isSacrificingCard.isTrue ? (
        <Header>Choose Card to Sacrifice</Header>
      ) : null}
      {isGivingToOpponent.isTrue ? <Header>Choose Card to Give</Header> : null}
      <Card.Group>
        {viewStablePlayer &&
          viewStablePlayer.stable &&
          currentPlayer.stable.map((card, index) => {
            return (
              <CardComponent
                index={index}
                key={card.id}
                card={card}
                callback={handleCallback}
              />
            );
          })}
      </Card.Group>
    </div>
  );
});

export default MemoStableComponent;
