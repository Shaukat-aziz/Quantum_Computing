import React from "react";
import { Segment, Header, Label, Card, Image } from "semantic-ui-react";
import "./ModalComponent.scss";

// Components
import CardComponent from "components/Card/CardComponent";

const ModalComponent = React.memo((props) => {
  const { header, cards, close, callback } = props;

  return (
    <Segment id="modal-component">
      <Header>{Header}</Header>
      <Label
        icon="close"
        corner="right"
        color="red"
        onClick={() => {
          close("");
        }}
      />
      {cards.map((card, index) => (
        <CardComponent
          index={index}
          key={card.id}
          card={card}
          callback={callback ? callback : () => {}}
        />
      ))}
    </Segment>
  );
});

export default ModalComponent;
