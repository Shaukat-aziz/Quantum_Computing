import React from "react";
import { Card, Popup, Image } from "semantic-ui-react";
import "./CardComponent.css";

const MemoCardComponent = React.memo((props) => {
  const { card, index, callback } = props;

  return (
    <Popup
      inverted
      trigger={
        <Card
          onClick={() => {
            callback(card, index);
          }}
        >
          <Image src={card.url} />
          <Card.Content>
            <Card.Header>{card.name}</Card.Header>
          </Card.Content>
        </Card>
      }
    >
      <Popup.Header>{card.type}</Popup.Header>
      <Popup.Content>{card.description}</Popup.Content>
    </Popup>
  );
});

export default MemoCardComponent;
