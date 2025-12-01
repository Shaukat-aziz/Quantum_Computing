import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { Segment, Card, Image } from "semantic-ui-react";
import { viewStable, toggleViewingOtherPlayerModal } from "actions";
import "./PlayersView.scss";
import { useMyPlayer } from "utils/hooks.js";

// Components
import ModalComponent from "components/Modal/ModalComponent";

function PlayersView() {
  const currentPlayer = useMyPlayer();
  const [selectedPlayer, setSelectedPlayer] = useState(false);
  const [playerHovered, setPlayerHover] = useState({});
  const game = useSelector((state) => state.game);
  const players = useSelector((state) => state.players);
  const dispatch = useDispatch();

  useEffect(() => {
    if (!selectedPlayer) return;

    if (currentPlayer.id == selectedPlayer.id) {
      // This happens when you click yourself
      dispatch(viewStable(currentPlayer, null));
    } else if (game.whosTurn.id == currentPlayer.id) {
      // Only the player whos turn it is should be able to view a hand/stable
      dispatch(toggleViewingOtherPlayerModal(currentPlayer, selectedPlayer.id));
    } else {
      // This is hit when a player whos turn its not clicks a stable.
      dispatch(viewStable(currentPlayer, selectedPlayer));
    }
  }, [currentPlayer, dispatch, game.whosTurn.id, selectedPlayer]);

  // useEffect(() => {
  //   if (currentPlayer.id) {
  //     setPlayerHover({...currentPlayer, index: 1});
  //     setShowQuickView(true);
  //   }
  // }, [currentPlayer])

  function toggleQuickView(player, index) {
    setPlayerHover({
      ...player,
      index,
    });
  }

  function renderQuickView() {
    if (playerHovered.id) {
      return (
        <QuickViewComponent
          stable={playerHovered.stable}
          index={playerHovered.index}
        />
      );
    }
  }

  return (
    <div id="players-view">
      <Card.Group itemsPerRow={1}>
        {players.map((player, index) => {
          return (
            <Card raised id={`playercard-${index}`} key={player.id}>
              <Image
                onClick={() => {
                  setSelectedPlayer(player);
                }}
                onMouseEnter={() => {
                  toggleQuickView(player, index);
                }}
                onMouseLeave={() => {
                  toggleQuickView({});
                }}
                label={{
                  color: player.color,
                  content: `${player.name}: H: ${player.hand.length} S: ${player.stable.length}`,
                  ribbon: true,
                }}
                src={player.unicorn.url}
              />
            </Card>
          );
        })}
        {renderQuickView()}
      </Card.Group>
    </div>
  );
}

function QuickViewComponent(props) {
  const { stable, index } = props;
  const blankCards = [{}, {}, {}, {}, {}, {}, {}];
  const MAX_CARDS = 7;
  const cardPosition = document
    .getElementById(`playercard-${index}`)
    .getBoundingClientRect();

  return (
    <Segment
      inverted
      style={{
        left: `${cardPosition.x + 150}px`,
        top: `${cardPosition.top - 40}px`,
      }}
    >
      {stable.map((card) => {
        return (
          <Card raised key={card.id}>
            <Image
              style={{ height: `${cardPosition.height}px` }}
              src={card.url}
            />
          </Card>
        );
      })}

      {blankCards.map((card, index) => {
        if (index < MAX_CARDS - stable.length) {
          return (
            <Card raised key={index}>
              <Image
                style={{ height: `${cardPosition.height}px` }}
                src={`https://unstableunicornsgame.s3.us-east-2.amazonaws.com/pngs/cardBack.jpg`}
              />
            </Card>
          );
        }
      })}
    </Segment>
  );
}

export default PlayersView;
