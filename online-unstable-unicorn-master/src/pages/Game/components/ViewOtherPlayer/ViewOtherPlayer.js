import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { Button, Modal } from "semantic-ui-react";
import { viewStable, toggleViewingOtherPlayerModal } from "actions";
import { useViewingPlayer, useMyPlayer } from "utils/hooks.js";

function ViewOtherPlayer(props) {
  const dispatch = useDispatch();
  const currentPlayer = useMyPlayer();
  const playerToView = useViewingPlayer();
  const [isViewingHand, setisViewingHand] = useState(false);
  function viewHand() {
    setisViewingHand(true);
  }

  function viewStableModal(selectedPlayer) {
    dispatch(viewStable(currentPlayer, selectedPlayer));
    dispatch(toggleViewingOtherPlayerModal(currentPlayer, currentPlayer.id));
  }

  function close() {
    setisViewingHand(false);
    dispatch(toggleViewingOtherPlayerModal(currentPlayer, currentPlayer.id));
  }

  return (
    <Modal open={props.isOpen} closeOnEscape={false} closeOnDimmerClick={false}>
      <Modal.Header>View {playerToView.name}'s </Modal.Header>
      {isViewingHand && (
        <Modal.Content>
          {playerToView.hand.map((card) => {
            return <p key={card.id}>{card.name}</p>;
          })}
        </Modal.Content>
      )}
      {!isViewingHand && (
        <Modal.Actions center="true">
          <Button
            onClick={() => {
              viewStableModal(playerToView);
            }}
          >
            Stable
          </Button>
          <Button onClick={viewHand} content="Hand" />
        </Modal.Actions>
      )}
      {isViewingHand && (
        <Modal.Actions center="true">
          <Button onClick={close}>Close</Button>
        </Modal.Actions>
      )}
    </Modal>
  );
}

export default ViewOtherPlayer;
