import { combineReducers } from "redux";
import socketIOClient from "socket.io-client";
import cards from "../db/cards.js";
import upgrades from "../db/upgrades.js";
import downgrades from "../db/downgrades.js";
import GroupBy from "lodash/groupBy";

let defaultOptions = {
  gameid: 0,
  gameDuration: "",
  expansion: "",
  winCondition: "",
  playing: false,
  gameOver: false,
  phase: 0,
  phases: [
    {
      name: "Pre Effects",
      actions: ["checkForEffects"],
    },
    {
      name: "Draw",
      actions: ["drawCard"],
    },
    {
      name: "Action",
      actions: ["playCard", "drawCard"],
    },
    {
      name: "Post Effects",
      actions: ["endTurn"],
    },
  ],
  cards,
  upgrades,
  downgrades,
  whosTurn: {},
  turn: 0,
};
const socketServer = socketIOClient("http://127.0.0.1:4404");

function socket(state = socketServer, action) {
  switch (action.type) {
    default:
      return state;
  }
}

function currentPlayerIndex(state = null, action) {
  switch (action.type) {
    case "SET_PLAYER":
      return action.player;
    default:
      return state;
  }
}

function game(state = defaultOptions, action) {
  let game = { ...state };
  switch (action.type) {
    case "JOIN_LOBBY":
      return {
        ...state,
        roomId: action.lobby.key,
        uri: action.lobby.name,
      };

    case "LEAVE_LOBBY":
      return defaultOptions;

    case "START_GAME":
      return {
        ...state,
        ...action.options,
      };

    case "NEXT_PHASE":
    case "END_ACTION_PHASE":
      console.log("calling next phase");
      game.phase = action.newPhase;

      return game;

    case "END_TURN":
      return {
        ...game,
        ...action.gameUpdates,
      };

    case "END_GAME":
      return {
        ...game,
        gameOver: true,
      };

    default:
      return state;
  }
}

function decks(
  state = {
    drawPile: [],
    nursery: GroupBy(cards, "type")["Baby Unicorn"],
    discardPile: [],
  },
  action
) {
  switch (action.type) {
    case "UPDATE_DECKS":
    case "START_GAME":
      return action.decks;

    case "UPDATE_DRAWPILE":
      return (decks.drawPile = action.deck);

    case "UPDATE_NURSERY":
      return (decks.nursery = action.deck);

    case "UPDATE_DISCARDPILE":
      return (decks.discardPile = action.deck);

    case "DRAW_CARD":
      break;
    default:
      return state;
  }
}

function isMyTurn(state = false, action) {
  switch (action.type) {
    case "START_GAME":
    case "END_TURN":
      return action.isMyTurn;

    default:
      return state;
  }
}

function isPlayingCard(
  state = { isTrue: false, callback: null, basicUnicornOnly: false },
  action
) {
  switch (action.type) {
    case "PLAYING_CARD":
      return action.isPlayingCard;

    case "END_ACTION_PHASE":
      return { isTrue: false, callback: null, basicUnicornOnly: false };

    default:
      return state;
  }
}

function isDiscardingCard(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "DISCARDING_CARD":
      return action.isDiscardingCard;

    default:
      return state;
  }
}

function isDiscardingOpponentCard(
  state = { isTrue: false, type: null, callback: null },
  action
) {
  switch (action.type) {
    case "DISCARDING_OPPONENT_CARD":
      return action.isDiscardingOpponentCard;

    default:
      return state;
  }
}

function isSacrificingCard(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "SACRIFICING_CARD":
      return action.isSacrificingCard;

    default:
      return state;
  }
}

function isDestroyingCard(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "DESTROYING_CARD":
      return action.isDestroyingCard;

    default:
      return state;
  }
}

function isReturningCard(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "RETURNING_CARD":
      return action.isReturningCard;

    default:
      return state;
  }
}

function isReturningOpponentCard(
  state = { isTrue: false, callback: null },
  action
) {
  switch (action.type) {
    case "RETURNING_OPPONENT_CARD":
      return action.isReturningOpponentCard;

    default:
      return state;
  }
}

function isDrawingFromOpponent(
  state = { isTrue: false, callback: null },
  action
) {
  switch (action.type) {
    case "DRAWING_FROM_OPPONENT":
      return action.isDrawingFromOpponent;

    default:
      return state;
  }
}

function isGivingToOpponent(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "GIVING_TO_OPPONENT":
      return action.isGivingToOpponent;

    default:
      return state;
  }
}

function isStealingUnicorn(state = { isTrue: false, callback: null }, action) {
  switch (action.type) {
    case "STEALING_UNICORN":
      return action.isStealingUnicorn;

    default:
      return state;
  }
}

function isChoosingPlayer(
  state = { isTrue: false, card: null, callback: null },
  action
) {
  switch (action.type) {
    case "CHOOSING_PLAYER":
      return action.isChoosingPlayer;

    default:
      return state;
  }
}

function cardBeingPlayed(state = {}, action) {
  switch (action.type) {
    case "ATTEMPT_ADD_TO_STABLE":
      return action.card;

    case "END_ACTION_PHASE":
      return {};

    default:
      return state;
  }
}

function players(state = [], action) {
  let newState;
  let currIndex;
  switch (action.type) {
    case "SET_PLAYERS":
    case "START_GAME":
      return action.players;

    case "LEAVE_LOBBY":
      return [];

    case "VIEW_STABLE":
      newState = [].concat(state);
      currIndex = newState.findIndex(
        (player) => player.id == action.currentPlayer.id
      );
      if (currIndex < 0) return state;

      const viewingStableId =
        action.viewingPlayer != null
          ? action.viewingPlayer.id
          : action.currentPlayer.id;
      action.currentPlayer.viewingStableId = viewingStableId;
      newState[currIndex] = action.currentPlayer;
      return newState;

    case "VIEW_OTHER_PLAYER_MODAL":
      newState = [].concat(state);
      currIndex = newState.findIndex(
        (player) => player.id == action.currentPlayer.id
      );
      if (currIndex < 0) return state;

      action.currentPlayer.viewingOtherPlayerModalId =
        action.viewingOtherPlayerModalId;
      newState[currIndex] = action.currentPlayer;
      return newState;

    default:
      return state;
  }
}

const rootReducer = combineReducers({
  socket,
  currentPlayerIndex,
  game,
  isMyTurn,
  isPlayingCard,
  isDiscardingCard,
  isDiscardingOpponentCard,
  isSacrificingCard,
  isDestroyingCard,
  isReturningCard,
  isReturningOpponentCard,
  isDrawingFromOpponent,
  isGivingToOpponent,
  isStealingUnicorn,
  isChoosingPlayer,
  players,
  decks,
  cardBeingPlayed,
});

export default rootReducer;
