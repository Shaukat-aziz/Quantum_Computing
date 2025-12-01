const cards = require("../cards.js");

module.exports = [
  {
    connected: true,
    color: "purple",
    name: "tyler",
    hand: [
      {
        ...cards[7],
        name: "Malik's the best",
        url:
          "https://unstableunicornsgame.s3.us-east-2.amazonaws.com/pngs/46.png",
        activateOnPlay: 1,
      },
    ],
    stable: [cards[7]],
    viewingStableId: 1,
    viewingOtherPlayerModalId: 1,
    unicorn: cards[7],
    upgrades: [],
    downgrades: [
      {
        name: "skipPhase",
        description: "Skip either your Draw phase or your Action phase.",
        optional: false,
      },
    ],
  },
  {
    connected: true,
    name: "Malik",
    color: "blue",
    hand: [],
    stable: [cards[3]],
    viewingStableId: 2,
    viewingOtherPlayerModalId: 2,
    unicorn: cards[3],
    upgrades: [],
    downgrades: [],
  },
  {
    connected: true,
    name: "Liz",
    color: "teal",
    hand: [],
    stable: [cards[12]],
    viewingStableId: 3,
    viewingOtherPlayerModalId: 3,
    unicorn: cards[12],
    upgrades: [],
    downgrades: [],
  },
  {
    connected: true,
    color: "green",
    name: "Troy",
    hand: [],
    stable: [cards[6]],
    viewingStableId: 4,
    viewingOtherPlayerModalId: 4,
    unicorn: cards[6],
    upgrades: [],
    downgrades: [],
  },
];
