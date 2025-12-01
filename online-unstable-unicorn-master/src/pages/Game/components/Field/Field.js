import React, { useState } from "react";
import { useSelector } from "react-redux";
import { Card, Image } from "semantic-ui-react";
import "./Field.css";

// Components
import ModalComponent from "components/Modal/ModalComponent";

const MemoField = React.memo(Field);

function Field() {
  const [isViewingDeck, setIsViewingDeck] = useState(false);
  const [deckBeingViewed, setDeckBeingViewed] = useState("");
  const decks = useSelector((state) => state.decks);

  function toggleViewDeckModal(id) {
    if (id !== "drawPile") {
      setIsViewingDeck(!isViewingDeck);
      setDeckBeingViewed(id);
    }
  }

  function renderViewDeckModal() {
    if (isViewingDeck) {
      return (
        <ModalComponent
          header="Cards left"
          cards={decks[deckBeingViewed]}
          close={toggleViewDeckModal}
        />
      );
    }
  }

  return (
    <div className="field">
      {Object.keys(decks).map((deckKey) => {
        return (
          <MemoDeck
            id={deckKey}
            key={deckKey}
            numCards={decks[deckKey].length}
            callback={toggleViewDeckModal}
          />
        );
      })}

      {renderViewDeckModal()}
    </div>
  );
}

const MemoDeck = React.memo(Deck);

function Deck(props) {
  const { id, numCards, callback } = props;

  return (
    <Card
      raised
      onClick={() => {
        callback(id);
      }}
    >
      <Image
        label={{
          color: "black",
          content: `${id}: ${numCards}`,
          ribbon: true,
        }}
        src={`https://unstableunicornsgame.s3.us-east-2.amazonaws.com/pngs/cardBack.jpg`}
      />
    </Card>
  );
}

export default MemoField;
