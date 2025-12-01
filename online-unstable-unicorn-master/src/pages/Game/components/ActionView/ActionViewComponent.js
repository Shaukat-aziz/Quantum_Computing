import React, { useEffect, useCallback } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  updateDecks,
  setPlayers,
  endActionPhase,
  discardingCard,
  sacrificingCard,
} from "actions";
import "./ActionViewComponent.scss";
import { Segment, Step } from "semantic-ui-react";

// Components
import PlayerView from "./components/PlayerView/Playerview.js";
import SpectatorView from "./components/SpectatorView/SpectatorView.js";

function ActionViewComponent() {
  const isMyTurn = useSelector((state) => state.isMyTurn);
  const socketServer = useSelector((state) => state.socket);
  const lobbyName = useSelector((state) => state.game.uri);
  const currentPlayerIndex = useSelector((state) => state.currentPlayerIndex);
  const players = useSelector((state) => state.players);
  const dispatch = useDispatch();

  const dispatchUpdates = useCallback(
    (updatedDecks, updatedPlayers) => {
      dispatch(updateDecks(updatedDecks));
      dispatch(setPlayers(updatedPlayers));
    },
    [dispatch]
  );

  useEffect(() => {
    socketServer.on("cardDrew", (updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("cardDiscarded", (card, updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("cardDestroyed", (card, updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("cardSacrificed", (card, updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("cardReturned", (card, updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on(
      "cardDrewFromOpponent",
      (card, updatedDecks, updatedPlayers) => {
        dispatchUpdates(updatedDecks, updatedPlayers);
      }
    );

    socketServer.on(
      "cardGivenToPlayer",
      (card, updatedDecks, updatedPlayers) => {
        dispatchUpdates(updatedDecks, updatedPlayers);
      }
    );

    socketServer.on("unicornStolen", (card, updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("setPlayersDiscarding", (playerIndexes) => {
      for (var i = 0; i < playerIndexes.length; i++) {
        if (
          players[parseInt(currentPlayerIndex)].id ===
          players[playerIndexes[i]].id
        ) {
          dispatch(
            discardingCard({
              isTrue: true,
              callback: () => {
                socketServer.emit(
                  "discardCheck",
                  lobbyName,
                  currentPlayerIndex
                );
              },
            })
          );
        }
      }
    });

    socketServer.on("setPlayersSacrificing", (playerIndexes) => {
      for (var i = 0; i < playerIndexes.length; i++) {
        if (
          players[parseInt(currentPlayerIndex)].id ===
          players[playerIndexes[i]].id
        ) {
          dispatch(
            sacrificingCard({
              isTrue: true,
              callback: () => {
                socketServer.emit(
                  "sacrificeCheck",
                  lobbyName,
                  currentPlayerIndex
                );
              },
            })
          );
        }
      }
    });

    socketServer.on("updateFromAction", (updatedDecks, updatedPlayers) => {
      dispatchUpdates(updatedDecks, updatedPlayers);
    });

    socketServer.on("endingActionPhase", () => {
      dispatch(endActionPhase());
    });

    return () => {
      socketServer.removeListener("cardDrew");
      socketServer.removeListener("cardDiscarded");
      socketServer.removeListener("cardDestroyed");
      socketServer.removeListener("cardSacrificed");
      socketServer.removeListener("cardReturned");
      socketServer.removeListener("cardDrewFromOpponent");
      socketServer.removeListener("cardGivenToPlayer");
      socketServer.removeListener("unicornStolen");
      socketServer.removeListener("updateFromAction");
      socketServer.removeListener("setPlayersDiscarding");
      socketServer.removeListener("setPlayersSacrificing");
      socketServer.removeListener("endingActionPhase");
    };
  }, [
    currentPlayerIndex,
    dispatch,
    dispatchUpdates,
    lobbyName,
    players,
    socketServer,
  ]);

  return (
    <div id="actionView">
      <MemoPhase />
      <Segment raised attached>
        {isMyTurn ? <PlayerView /> : <SpectatorView />}
      </Segment>
    </div>
  );
}

const MemoPhase = React.memo(() => {
  const phases = useSelector((state) => state.game.phases);
  const currentPhase = useSelector((state) => state.game.phase);

  return (
    <Step.Group attached="top" fluid>
      {phases.map((phase) => (
        <Step
          key={phase.name}
          active={phases[currentPhase].name === phase.name}
          disabled={false}
        >
          <Step.Content>
            <Step.Title>{phase.name}</Step.Title>
            {/*
              TODO: show number of actions per phase
              <Step.Description>Choose your shipping options</Step.Description>
            */}
          </Step.Content>
        </Step>
      ))}
    </Step.Group>
  );
});

export default ActionViewComponent;
