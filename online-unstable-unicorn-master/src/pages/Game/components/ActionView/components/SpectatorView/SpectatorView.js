import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { nextPhase, endTurn } from "actions";
import { useMyPlayer, useCheckForInstants } from "utils/hooks.js";
import { Button } from "semantic-ui-react";

// Components
import CardComponent from "components/Card/CardComponent";

const MemoSpectatorView = React.memo(() => {
  const myPlayer = useMyPlayer();
  const [card, setCard] = useState(null);
  const socketServer = useSelector((state) => state.socket);
  const currentGame = useSelector((state) => state.game);
  const dispatch = useDispatch();
  const [checkForInstant, setCheckForInstant] = useState(false);

  useEffect(() => {
    socketServer.on("switchingPhase", (phase) => {
      dispatch(nextPhase(phase));
    });

    socketServer.on("attemptCardPlay", (card, updatedPlayers) => {
      setCard(card);
      setCheckForInstant(true);
    });

    socketServer.on("endingTurn", (gameUpdates, nextPlayerIndex) => {
      dispatch(
        endTurn(
          gameUpdates,
          nextPlayerIndex ===
            parseInt(localStorage.getItem("currentPlayerIndex"))
        )
      );
    });

    return () => {
      socketServer.removeListener("switchingPhase");
      socketServer.removeListener("attemptCardPlay");
      socketServer.removeListener("endingTurn");
    };
  }, [dispatch, socketServer]);

  function handleInstant(instant) {
    //TODO: make one call on there socket server
    if (instant.name === "Skip") {
      socketServer.emit(
        "skippingInstant",
        currentGame.uri,
        myPlayer.currentPlayerIndex,
        card
      );
    } else {
      socketServer.emit(
        "playInstant",
        currentGame.uri,
        myPlayer.currentPlayerIndex,
        instant
      );
    }
    setCard(null);
    setCheckForInstant(false);
  }

  return (
    <div>
      {currentGame.whosTurn.name} Turn
      {card ? (
        <MemoCounterActionView
          card={card}
          checkForInstant={checkForInstant}
          handleInstant={handleInstant}
        />
      ) : null}
    </div>
  );
});

const MemoCounterActionView = React.memo((props) => {
  const { card, checkForInstant, handleInstant } = props;
  const instantActions = useCheckForInstants();

  function renderInstants() {
    if (checkForInstant) {
      return (
        <div>
          {instantActions.map((action) => {
            return (
              <Button
                key={action.id}
                content={action.name}
                onClick={() => handleInstant(action)}
              />
            );
          })}
        </div>
      );
    }
  }

  return (
    <div>
      {renderInstants()}
      <CardComponent card={card} callback={() => {}} />
    </div>
  );
});

export default MemoSpectatorView;
