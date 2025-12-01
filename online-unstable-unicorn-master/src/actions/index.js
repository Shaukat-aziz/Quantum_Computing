// Socket Actions
export const joinLobby = (lobby) => ({
  type: "JOIN_LOBBY",
  lobby,
});

export const leaveLobby = () => ({
  type: "LEAVE_LOBBY",
});

// Current player actions
export const setCurrentPlayer = (player) => ({
  type: "SET_PLAYER",
  player,
});

// Game actions
export const startGame = (options, decks, players, isMyTurn) => ({
  type: "START_GAME",
  options,
  decks,
  players,
  isMyTurn,
});

export const nextPhase = (newPhase) => ({
  type: "NEXT_PHASE",
  newPhase,
});

export const viewStable = (currentPlayer, viewingPlayer) => ({
  type: "VIEW_STABLE",
  currentPlayer,
  viewingPlayer,
});

export const toggleViewingOtherPlayerModal = (
  currentPlayer,
  viewingOtherPlayerModalId
) => ({
  type: "VIEW_OTHER_PLAYER_MODAL",
  currentPlayer,
  viewingOtherPlayerModalId,
});

export const endActionPhase = () => ({
  type: "END_ACTION_PHASE",
  newPhase: 3,
});

export const endTurn = (gameUpdates, isMyTurn) => ({
  type: "END_TURN",
  gameUpdates,
  isMyTurn,
});

// Deck Actions
export const updateDecks = (decks) => ({
  type: `UPDATE_DECKS`,
  decks,
});

export const updateHand = (player) => ({
  type: "UPDATE_HAND",
  player,
});

// Player actions
export const setPlayers = (players) => ({
  type: "SET_PLAYERS",
  players,
});

export const playCard = ({ users, options }) => ({
  type: "START_GAME",
  options,
});

export const playingCard = (isPlayingCard) => ({
  type: "PLAYING_CARD",
  isPlayingCard,
});

export const discardCard = (updatedDecks, updatedPlayers) => ({
  type: "DISCARD_CARD",
  updatedDecks,
  updatedPlayers,
});

export const discardingCard = (isDiscardingCard) => ({
  type: "DISCARDING_CARD",
  isDiscardingCard,
});

export const discardingOpponentCard = (isDiscardingOpponentCard) => ({
  type: "DISCARDING_OPPONENT_CARD",
  isDiscardingOpponentCard,
});

export const sacrificingCard = (isSacrificingCard) => ({
  type: "SACRIFICING_CARD",
  isSacrificingCard,
});

export const destroyingCard = (isDestroyingCard) => ({
  type: "DESTROYING_CARD",
  isDestroyingCard,
});

export const returningCard = (isReturningCard) => ({
  type: "RETURNING_CARD",
  isReturningCard,
});

export const returningOpponentCard = (isReturningOpponentCard) => ({
  type: "RETURNING_OPPONENT_CARD",
  isReturningOpponentCard,
});

export const drawingFromOpponent = (isDrawingFromOpponent) => ({
  type: "DRAWING_FROM_OPPONENT",
  isDrawingFromOpponent,
});

export const givingToOpponent = (isGivingToOpponent) => ({
  type: "GIVING_TO_OPPONENT",
  isGivingToOpponent,
});

export const stealUnicorn = (isStealingUnicorn) => ({
  type: "STEALING_UNICORN",
  isStealingUnicorn,
});

export const choosePlayer = (isChoosingPlayer) => ({
  type: "CHOOSING_PLAYER",
  isChoosingPlayer,
});

export const attemptToPlay = (card) => ({
  type: "ATTEMPT_ADD_TO_STABLE",
  card,
});

export const endGame = (updatedGame) => ({
  type: "END_GAME",
  updatedGame,
});
