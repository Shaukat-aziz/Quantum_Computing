const upgrades = [
  {
    id: 1,
    name: "sacrificeAndDestroy",
    description: "You may SACRIFICE a card. If you do, DESTROY a card.",
    optional: true,
  },
  {
    id: 2,
    name: "doubleActions",
    description: "You may play 2 cards during your action phase.",
    optional: true,
  },
  {
    id: 3,
    name: "drawFromPlayerRandom",
    description:
      "You may choose any other player. Pull a card from that player's hand and add it to your hand. If you do, skip your Draw phase.",
    optional: true,
  },
  {
    id: 4,
    name: "borrowUnicorn",
    description:
      "You may STEAL a Unicorn card. At the end of your turn, return that Unicorn card to the Stable from which you stole it.",
    optional: true,
  },
  {
    id: 5,
    name: "drawExtraCard",
    description: "You may DRAW an extra card.",
    optional: true,
  },
  {
    id: 6,
    name: "swapUnicorns",
    description:
      "You may move a Unicorn card from any player's Stable to any other player's Stable. You cannot move that card into your own Stable.",
    optional: true,
  },
  {
    id: 7,
    name: "basicBitch",
    description:
      "You may bring a Basic Unicorn card from your hand directly into your Stable.",
    optional: true,
  },
  {
    id: 8,
    name: "volunteerAsTribute",
    description:
      "If 1 of your Unicorns would be sacrificed or destroyed, you may SACRIFICE this card instead.",
    optional: true,
  },
  {
    id: 9,
    name: "unKillable",
    description: "Your Unicorn cards cannot be destroyed.",
    optional: false,
  },
  {
    id: 10,
    name: "drawBeforeEnding",
    description: "You may DRAW a card at the end of your turn.",
    optional: true,
  },
  {
    id: 11,
    name: "sacrificeBaby",
    description:
      "You may SACRIFICE a Basic Unicorn card. If you do, pull a card from each other player's hand and add it to your hand.",
    optional: true,
  },
  {
    id: 12,
    name: "getBabyUnicornInStable",
    description:
      "Bring a Baby Unicorn card from the Nursery directly into your Stable.",
    optional: true,
  },
  {
    id: 13,
    name: "farSight",
    description: "See the top 3 cards on deck",
    optional: false,
  },
  {
    id: 14,
    name: "findBearDaddy",
    description:
      'you may search the deck for a "Bear Daddy Unicorn" card. Add it to your hand, then shuffle the deck.',
    optional: true,
  },
  {
    id: 15,
    name: "drawFromDiscardPile",
    description:
      "When this card enters your Stable, you may choose a card from the discard pile and add it to your hand.",
    optional: false,
  },
  {
    id: 16,
    name: "sacrificeHand",
    description:
      "If this card is in your Stable at the beginning of your turn, you may DISCARD your hand. If you do, DESTROY a Unicorn card.",
    optional: true,
  },
  {
    id: 17,
    name: "increaseHandLimit",
    description: "Your hand limit increases by 1.",
    optional: false,
  },
  {
    id: 18,
    name: "holdMyUnicorn",
    description:
      "If this card is in your Stable at the beginning of your turn, move a Unicorn card from your Stable into any other player's Stable. At the end of your turn, return that Unicorn card to your Stable.",
    optional: false,
  },
  {
    id: 19,
    name: "findTwinkicorn",
    description:
      'When this card enters your Stable, you may search the deck for a "Twinkicorn" card. Add it to your hand, then shuffle the deck.',
    optional: true,
  },
  {
    id: 20,
    name: "makePlayerDiscard",
    description:
      "When this card enters your Stable, you may choose any player. That player must DISCARD a card. If this card would be sacrificed or destroyed, return it to your hand instead.",
    optional: true,
  },
  {
    id: 21,
    name: "stealUnicorn",
    description:
      "If this card is in your Stable at the beginning of your turn, you may move this card to any other player's Stable. If you do, STEAL a Unicorn card from that player's Stable.",
    optional: false,
  },
  {
    id: 22,
    name: "getBabyUnicornInHand",
    description:
      "When this card enters your Stable, you may DISCARD a card from the Nursery directly into your Stable.",
    optional: false,
  },
  {
    id: 23,
    name: "makePlayerReturnCard",
    description:
      "When this card enters your Stable, you may return a card in any player's Stable to that player's hand.",
    optional: false,
  },
  {
    id: 24,
    name: "stealBasicUnicorn",
    description:
      "When this card enters your Stable, STEAL a Basic Unicorn card. If this card leaves your Stable, return that Basic Unicorn card to the Stable from which you stole it.",
    optional: false,
  },
  {
    id: 25,
    name: "noBabyUnicorns",
    description: "Baby Unicorn cards cannot enter any player's Stable.",
    optional: false,
  },
  {
    id: 26,
    name: "discardPlayerCard",
    description:
      "When this card enters your Stable, choose a player and look at their hand. Choose a Unicorn card in that player's hand and move it to the discard pile.",
    optional: false,
  },
  {
    id: 27,
    name: "everyoneSacrifice",
    description:
      "When this card enters your Stable, each player must SACRIFICE a Unicorn card.",
    optional: false,
  },
  {
    id: 28,
    name: "chooseFromDeck",
    description:
      "When this card enters your Stable, you may search the deck for a Unicorn card and add it to your hand. Shuffle the deck.",
    optional: true,
  },
  {
    id: 29,
    name: "drawThenDiscard",
    description:
      "If this card is in your Stable at the beginning of your turn, you may DRAW an extra card, then DISCARD a card.",
    optional: true,
  },
  {
    id: 30,
    name: "sacrificeDowngrades",
    description:
      "When this card enters your Stable, SACRIFICE all Downgrade cards.",
    optional: false,
  },
  {
    id: 31,
    name: "twiceTheHorsePower",
    description:
      "This card counts for 2 Unicorns. You cannot play Basic Unicorn cards.",
    optional: false,
  },
  {
    id: 32,
    name: "stealBabyUnicornTemp",
    description:
      "When this card enters your Stable, STEAL a Baby Unicorn card. If this card leaves your Stable, return that Baby Unicorn card to the Stable from which you stole it.",
    optional: false,
  },
  {
    id: 33,
    name: "everyoneDiscard",
    description:
      "When this card enters your Stable, each other player must DISCARD a card.",
    optional: false,
  },
  {
    id: 34,
    name: "sacrificeAndDestroyHand",
    description:
      "Each player must SACRIFICE a card and DISCARD their hand. Shuffle the discard pile into the deck. Deal 5 cards to each player.",
    optional: false,
  },
  {
    id: 35,
    name: "skipPlayersTurn",
    description: "Choose any player. That player must skip their next turn.",
    optional: false,
  },
  {
    id: 36,
    name: "discardAndSacrifice",
    description:
      "Choose a player. That player must DISCARD 3 cards or SACRIFICE a Unicorn card.",
    optional: false,
  },
  {
    id: 37,
    name: "draw2FromPlayer",
    description:
      "Choose any other player. Pull 2 cards from that player's hand and add them to your hand.",
    optional: false,
  },
  {
    id: 38,
    name: "stealBabyUnicorn",
    description: "STEAL a Baby Unicorn card.",
    optional: false,
  },
  {
    id: 39,
    name: "discardDrawRefresh",
    description: "DISCARD 2 cards, then DRAW 2 cards. Take another turn.",
    optional: false,
  },
  {
    id: 40,
    name: "discardGiveDestroy",
    description:
      "Choose any player. That player must DISCARD a card. Give any player a card from your hand, then DESTROY a Unicorn card.",
    optional: false,
  },
  {
    id: 41,
    name: "getRidOfDowngrades",
    description:
      "SACRIFICE or DESTROY all Downgrade cards in any player's Stable.",
    optional: false,
  },
  {
    id: 42,
    name: "removeUpgradeOrDowngrade",
    description: "SACRIFICE or DESTROY an Upgrade or Downgrade card.",
    optional: false,
  },
  {
    id: 43,
    name: "specialDraw",
    description:
      "Reveal the top card in the deck. If it is a Unicorn card, bring it directly into your Stable and take another turn. If it is a Downgrade card, DISCARD it and SACRIFICE a Unicorn card. If it is anything else, add it to your hand.",
    optional: false,
  },
  {
    id: 44,
    name: "returnBabyUnicorns",
    description:
      "Each player must return a Baby Unicorn card from their Stable to the Nursery.",
    optional: false,
  },
  {
    id: 45,
    name: "refreshHands",
    description:
      "Each player must DISCARD their hand and DRAW the same number of cards. DRAW 2 additional cards.",
    optional: false,
  },
  {
    id: 46,
    name: "destroyAndDestroy",
    description: "DESTROY a Unicorn card. Each player must DISCARD a card.",
    optional: false,
  },
  {
    id: 47,
    name: "refreshMyHand",
    description: "DISCARD your hand, then DRAW 3 cards. Take another turn.",
    optional: false,
  },
  {
    id: 48,
    name: "AdditonalDraws",
    description:
      "If you have more than 3 Unicorns in your Stable, DRAW cards equal to the number of Unicorns in your Stable minus 3.",
    optional: false,
  },

  {
    id: 49,
    name: "drawFromPlayer",
    description:
      "Choose any player and look at that player's hand. Choose a card from that player's hand and add it to your hand.",
    optional: false,
  },
  {
    id: 50,
    name: "getRidOfUpgrades",
    description:
      "SACRIFICE or DESTROY all Upgrade cards in any player's Stable.",
    optional: false,
  },
];

export default upgrades;
