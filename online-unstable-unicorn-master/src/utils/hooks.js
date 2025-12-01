import { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { useHistory, useParams } from "react-router-dom";
import { setCurrentPlayer } from "actions";
import groupBy from "lodash/groupBy";

export function useMyServer() {
  const urlParams = useParams().id;

  const [socketServer, setMySocketServer] = useState({});

  return socketServer;
}

export function useViewingPlayer() {
  const players = useSelector((state) => state.players);
  const me = useMyPlayer();
  return {
    ...players.find((player) => player.id == me.viewingOtherPlayerModalId),
  };
}

export function useMyPlayer() {
  // TODO: make this render less times
  const [myPlayer, setMyPlayer] = useState({
    hand: [],
    stable: [],
  });
  const currentPlayerIndex = useSelector((state) => state.currentPlayerIndex);
  const players = useSelector((state) => state.players);
  const dispatch = useDispatch();

  useEffect(() => {
    if (!currentPlayerIndex) {
      dispatch(setCurrentPlayer(localStorage.getItem("currentPlayerIndex")));
    }

    if (players[currentPlayerIndex]) {
      setMyPlayer({
        currentPlayerIndex,
        ...players[currentPlayerIndex],
      });
    }
  }, [players, currentPlayerIndex, dispatch]);

  return myPlayer;
}

export function useCheckForInstants() {
  const currentPlayerIndex = useSelector((state) => state.currentPlayerIndex);
  const myHand = useSelector((state) => state.players[currentPlayerIndex].hand);
  const [instantActions, setInstantActions] = useState([]);

  useEffect(() => {
    console.log("CHECKING FOR INSTANTS", myHand);
    let newActions = [];
    const cardTypes = groupBy(myHand, "type");
    if (cardTypes["Instant"]) {
      newActions = cardTypes["Instant"].map((instant) => {
        return {
          id: instant.id,
          name: instant.name,
        };
      });
    }

    setInstantActions([
      {
        id: 0,
        name: "Skip",
      },
      ...newActions,
    ]);
  }, [myHand]);

  return instantActions;
}
