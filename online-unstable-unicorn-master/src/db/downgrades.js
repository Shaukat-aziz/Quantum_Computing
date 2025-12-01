const downgrades = [
  {
    id: 1,
    name: "skipPhase",
    description: "Skip either your Draw phase or your Action phase.",
    optional: false,
  },
  {
    id: 2,
    name: "disableUnicornEffects",
    description: "Triggered effects of your Unicorn cards do not activate.",
    optional: false,
  },
  {
    id: 3,
    name: "noUpgrades",
    description: "You cannot play Upgrade cards.",
    optional: false,
  },
  {
    id: 4,
    name: "discardCard",
    description: "Discard a card",
    optional: false,
  },
  {
    id: 5,
    name: "returnToHand",
    description: "Return a Unicorn card",
    optional: false,
  },
  {
    id: 6,
    name: "limitHand",
    description: "Your hand limit is 3 cards",
    optional: false,
  },
  {
    id: 7,
    name: "horseShit",
    description:
      "All of your Unicorn cards are considered Shits. Cards that affect Unicorn cards do not affect Shits.",
    optional: false,
  },
  {
    id: 8,
    name: "stopCard",
    description:
      "Stop that player's card from being played and send it to the discard pile.",
    optional: false,
  },
  {
    id: 9,
    name: "stopAndDiscard",
    description:
      "Stop that player's card from being played and send it to the discard pile. That player must DISCARD a card.",
    optional: false,
  },
  {
    id: 10,
    name: "destroyCard",
    description: "Destroy a card",
    optional: false,
  },
  {
    id: 11,
    name: "discardCards",
    description: "Discard 2 card",
    optional: false,
  },
  {
    id: 12,
    name: "noMagicCards",
    description: "Cannot play magic cards",
    optional: false,
  },
  {
    id: 13,
    name: "farSight",
    description: "See the top 3 cards on deck",
    optional: false,
  },
];

export default downgrades;
