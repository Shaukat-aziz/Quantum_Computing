/* eslint-disable */ 
import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from 'react-redux';
import { setCurrentPlayer, setPlayers, startGame, nextPhase, endTurn, playingCard,
  discardingCard, discardingOpponentCard, returningCard, sacrificingCard, destroyingCard,
  drawingFromOpponent, givingToOpponent, choosePlayer, stealUnicorn, returningOpponentCard } from 'actions';
import { useMyPlayer, useCheckForInstants } from 'utils/hooks.js';
import groupBy from 'lodash/groupBy';
import Shuffle from 'lodash/shuffle';
import { Dropdown, Image, Item, Segment, Button, Header, Card } from 'semantic-ui-react';


// Components
import CardComponent from 'components/Card/CardComponent'
import ModalComponent from 'components/Modal/ModalComponent';

function PlayerView() {
  const myPlayer = useMyPlayer();
  const socketServer = useSelector(state => state.socket)
  const lobbyName = useSelector(state =>  state.game.uri);
  const isMyTurn = useSelector(state =>  state.isMyTurn);
  const players = useSelector(state =>  state.players);
  const game = useSelector(state =>  state.game);
  const decks = useSelector(state =>  state.decks);
  const cardBeingPlayed = useSelector(state =>  state.cardBeingPlayed);
  const isDiscardingCard = useSelector(state =>  state.isDiscardingCard);

  const isDestroyingCard = useSelector(state => state.isDestroyingCard);
  const isDrawingFromOpponent = useSelector(state => state.isDrawingFromOpponent);
  const isChoosingPlayer = useSelector(state =>  state.isChoosingPlayer);
  const isStealingUnicorn = useSelector(state =>  state.isStealingUnicorn);
  const isReturningOpponentCard = useSelector(state => state.isReturningOpponentCard);
  const isDiscardingOpponentCard = useSelector(state => state.isDiscardingOpponentCard);

  const dispatch = useDispatch();

  // Effect Phase variables
  const [effects, setEffects] = useState([]);
  const [numberOfEffectsTotal, setNumberOfEffectsTotal] = useState(null);
  const [numberOfEffectsHandled, setNumberOfEffectsHandled] = useState(0);
  const [allowSkip, setAllowSkip] = useState(true);

  // Number of turns in phases
  // const [numberOfEffectsLeft, setNumberOfEffectsLeft] = useState(null);
  const [numberOfDrawsLeft, setNumberOfDrawsLeft] = useState(1);
  const [numberOfActionsLeft, setNumberOfActionsLeft] = useState(1);
  const [numberOfEndingEffectsLeft, setNumberOfEndingEffectsLeft] = useState(0);

  // Action phase variables
  const [actionSideEffect, setActionSideEffect] = useState(null);

  // Handling Opponent Interaction variables
  const [numPlayersCheckedForInstants, setNumPlayersCheckedForInstants] = useState(0);
  const [opponentInteractions, setOpponentInteractions] = useState({
    numPlayersCheckedForInstants: 0,
    discardCard: false,
    checkForInstant: false,
    instant: null
  });
  const [checkForInstant, setCheckForInstant] = useState(false);

  const [numPlayersDiscarding, setNumPlayersDiscarding] = useState(null);
  const [discardingAction, setDiscardingAction] = useState(null);
  const [numPlayersSacrifcing, setNumPlayersSacrifcing] = useState(null);
  const [sacrificingAction, setSacrficingAction] = useState(null);

  // upgrade variables
  const [sacrificeAndDestroy, setSacrificeAndDestroy] = useState(false);
  const [drawFromOpponent, setDrawFromOpponent] = useState(false);
  const [borrowUnicorn, setBorrowUnicorn] = useState(false);
  const [returnCardAtEndOfTurn, setReturnCardAtEndOfTurn] = useState(null);
  const [drawDuringPostEffectPhase, setDrawDuringPostEffectPhase] = useState(null)

  // downgrade variables
  const [skipDrawPhase, setSkipDrawPhase] = useState(false);
  const [skipActionPhase, setSkipActionPhase] = useState(false);

  useEffect(() => {
    socketServer.on('playerCheckedForInstant', (playerIndex, instant) => {
      let interactions = opponentInteractions;
      if (instant) {
        console.log(`${players[playerIndex].name} played ${instant.name}`)
        interactions.instant = {
          ...instant,
          playerIndex
        }
        switch (instant.name) {
          case 'NEIGH MEANS NEIGH':
            interactions.checkForInstant = false;
            break;

          case 'NEIGH, MOTHERFUCKER':
            interactions.discardCard = true;
            interactions.checkForInstant = true;
            break;
          default:
            interactions.checkForInstant = true;
        }
      }

      interactions.numPlayersCheckedForInstants++
      setOpponentInteractions(interactions);
      setNumPlayersCheckedForInstants(interactions.numPlayersCheckedForInstants)
    })

    return () => {
      socketServer.removeListener('playerCheckedForInstant');
    }
  },[socketServer])

  useEffect(() => {
    console.log('opponentInteractions updated')
    console.log(opponentInteractions, players.length)
    console.log(numPlayersCheckedForInstants)
    if (opponentInteractions.instant) {
      if (opponentInteractions.checkForInstant) {
        console.log('checking for instant before ending turn');
        setCheckForInstant(true);
      } else {
        console.log('discarding card and ending turn');
        const [updatedDecks, updatedPlayers] = addToDiscardPile();
        if (game.phase === 2) {
          if (numberOfActionsLeft === 1) {
            socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
          }
          setNumberOfActionsLeft(numberOfActionsLeft - 1)
        } else {
          handleEffects(null, false);
          socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
        }
      }
    } else if (opponentInteractions.numPlayersCheckedForInstants === players.length - 1) {
      console.log('playing card and ending turn');
      const [updatedDecks, updatedPlayers] = addToStable();
      const callbackAction = game.phase === 2 ?
      () => {
        if (numberOfActionsLeft === 1) {
          socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
        }
        setNumberOfActionsLeft(numberOfActionsLeft - 1)
      } :
      () => {
        handleEffects(null, false);
        socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
      };

      cardBeingPlayed.activateOnPlay ? handleSideEffect(callbackAction) : callbackAction();
    }
  }, [numPlayersCheckedForInstants])

  useEffect(() => {
    if (discardingAction) {
      socketServer.on('playerCheckedForDiscarding', (playerIndex, instant) => {
        setNumPlayersDiscarding(numPlayersDiscarding - 1);
      })
    }

    return () => socketServer.removeListener('playerCheckedForDiscarding');
  })

  useEffect(() => {
    if (numPlayersDiscarding === 0) {
      discardingAction.callback()
    }
  }, [numPlayersDiscarding])

  useEffect(() => {
    if (sacrificingAction) {
      socketServer.on('playerCheckedForSacrifing', (playerIndex, instant) => {
        setNumPlayersSacrifcing(numPlayersSacrifcing - 1);
      })
    }

    return () => socketServer.removeListener('playerCheckedForSacrifing');
  })

  useEffect(() => {
    if (numPlayersSacrifcing === 0) {
      sacrificingAction.callback()
    }
  }, [numPlayersSacrifcing])

  useEffect(() => {
    if (game.phase === 0 && myPlayer.id && numberOfEffectsTotal === null) {
      console.log('STARTING EFFECT PHASE')
      const cardTypes = groupBy(myPlayer.stable, 'activateAtBeginning');
      if (cardTypes['true']) {
        let theEffects = [];
        setNumberOfEffectsTotal(cardTypes['true'].length);

        cardTypes['true'].forEach(card => {
          console.log(card.name);
          if (card.hasOwnProperty('upgrade')) {
            console.log(game.upgrades[card.upgrade])
            switch (card.upgrade) {
              case 0:
                // sacrifice then destroy card
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Sacrifice then Destory',
                    requiresAction: true,
                    callback: () => dispatch(sacrificingCard({
                      isTrue: true,
                      callback: () => dispatch(destroyingCard({
                        isTrue: true,
                        callback: () => handleEffects(null, false)
                      }))
                    }))
                  }]
                });
                break;
              case 1:
                // Play 2 actions
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Gain Additional Action',
                    callback: () => setNumberOfActionsLeft(numberOfActionsLeft + 1)
                  }]
                });
                break;
              case 2:
                // You may choose any other player. Pull a card from that player\'s hand and add it to your hand. If you do, skip your Draw phase.
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Draw from Opponent',
                    requiresAction: true,
                    callback: () => dispatch(drawingFromOpponent({
                      isTrue: true,
                      callback: () => handleEffects(() => { setNumberOfDrawsLeft(numberOfDrawsLeft -1) }, false)
                    }))
                  }]
                });
                break;
              case 3:
                // borrowUnicorn
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Borrow Unicorn',
                    requiresAction: true,
                    callback: () => dispatch(drawingFromOpponent({
                      isTrue: true,
                      callback: (card, playerIndex) => handleEffects(() => {
                        setReturnCardAtEndOfTurn({
                          ...card,
                          previousPlayerIndex: playerIndex,
                          currentPlayerIndex: myPlayer.currentPlayerIndex,
                          location: 'hand'
                        });
                        setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft + 1);
                      }, false)
                    }))
                  }]
                });
                break;
              case 4:
                // drawExtraCard
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Draw Additional Card',
                    callback: () => setNumberOfDrawsLeft(numberOfDrawsLeft + 1)
                  }]
                });
                break;
              case 5:
                // TODO: add this later
                // swapUnicorns -> requires 3 players
                break;
              case 6:
                // basicBitch
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Play Basic Unicorn',
                    requiresAction: true,
                    callback: () => dispatch(playingCard({
                      isTrue: true,
                      callback: () => handleEffects(() => {
                        dispatch(playingCard({
                          isTrue: false,
                          callback: null,
                          basicUnicornOnly: false
                        }))
                      }, false),
                      basicUnicornOnly: true
                    }))
                  }]
                });
                break;
              case 9:
                // drawBeforeEnding
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Draw during post effect phase',
                    callback: () => handleEffects(() => {
                      setDrawDuringPostEffectPhase(true);
                      setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft + 1);
                    }, false)
                  }]
                });
                break;
              case 11:
                // getBabyUnicornInStable
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Play Baby Unicorn from Nursery',
                    callback: () => handleEffects(() => {
                      summonBabyUnicorn();
                    }, false)
                  }]
                });
                break;
              case 15:
                // sacrificeHand
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Sacrifice Hand and destory card',
                    requiresAction: true,
                    callback: () => dispatch(destroyingCard({
                      isTrue: true,
                      callback: () => handleEffects(sacrificeHand, false)
                    }))
                  }]
                });
                break;
              case 17:
                // holdMyUnicorn
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Give Opponent A stable unicorn',
                    requiresAction: true,
                    callback: () => dispatch(givingToOpponent({
                      isTrue: true,
                      callback: (card) => dispatch(choosePlayer({
                        isTrue: true,
                        card,
                        callback: (playerIndex) => handleEffects(() => {
                          setReturnCardAtEndOfTurn({
                            ...card,
                            previousPlayerIndex: myPlayer.currentPlayerIndex,
                            currentPlayerIndex: playerIndex,
                            location: 'stable'
                          });
                          setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft + 1);
                        }, false)
                      }))
                    }))
                  }]
                });
                break;
              case 20:
                // stealUnicorn
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Steal Unicorn',
                    requiresAction: true,
                    callback: () => {
                      dispatch(choosePlayer({
                        isTrue: false,
                        card,
                        callback: null
                      }));

                      dispatch(stealUnicorn({
                        isTrue: true,
                        card: card,
                        callback: () => handleEffects(null, false)
                      }));
                    }
                  }]
                });
                break;
              case 28:
                // drawThenDiscard
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Draw and Destory',
                    requiresAction: true,
                    callback: () => dispatch(destroyingCard({
                      isTrue: true,
                      callback: () => {
                        drawCard(0);
                        handleEffects(null, false);
                      }
                    }))
                  }]
                });
                break;
            }
          }

          if (card.hasOwnProperty('downgrade')) {
            switch (card.downgrade) {
              case 0:
                // skip either your draw or action phase
                setAllowSkip(false);
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Skip Draw Phase',
                    requiresAction: false,
                    callback: () => setSkipDrawPhase(true)
                  },{
                    name: 'Skip Action Phase',
                    requiresAction: false,
                    callback: () => setSkipActionPhase(true)
                  }]
                });
                break;
              case 3:
                // discard a card
                setAllowSkip(false);
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Discard card',
                    requiresAction: true,
                    callback: () => dispatch(discardingCard({
                      isTrue: true,
                      callback: () => handleEffects(null, false)
                    }))
                  }]
                });
                break;
              case 4:
                // return a unicorn to hand
                // TODO: discard debuff when stable reaches 0 <----- v2
                setAllowSkip(false);
                theEffects.push({
                  name: card.name,
                  actions: [{
                    name: 'Return unicorn',
                    requiresAction: true,
                    callback: () => dispatch(returningCard({
                      isTrue: true,
                      callback: () => handleEffects(null, false)
                    }))
                  }]
                });
                break;
            }
          }
        })

        setEffects(theEffects);
        setNumberOfEffectsHandled(0);
      } else {
          setNumberOfEffectsTotal(0);
          setNumberOfEffectsHandled(0);
      }
    }

    if (game.phase === 1) {
      if (skipDrawPhase) {
        console.log('skipping')
        dispatch(nextPhase(2))
      }
    }

    if (game.phase === 2) {
      if (skipActionPhase) {
        dispatch(nextPhase(3))
      }
    }

    if (game.phase === 3 && !isDiscardingCard.isTrue) {
      console.log('STARTING END TURN PHASE')
      if (myPlayer.hand.length > 7) {
        dispatch(discardingCard({
          isTrue: true,
          callback: null
        }));
      } else {
        setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft - 1);
      }

      if (returnCardAtEndOfTurn) {
        const playerHoldingCard = players[returnCardAtEndOfTurn.currentPlayerIndex];
        let returnCardIndex = playerHoldingCard.hand.findIndex(card => card.id === returnCardAtEndOfTurn.id);
        if (returnCardIndex >= 0) {
          returnCardToOpponent('hand', returnCardIndex, returnCardAtEndOfTurn);
        } else {
          returnCardIndex = playerHoldingCard.stable.findIndex(card => card.id === returnCardAtEndOfTurn.id);
          if (returnCardIndex >= 0) {
            returnCardToOpponent('stable', returnCardIndex, returnCardAtEndOfTurn);
          } else {
            setReturnCardAtEndOfTurn(null);
            setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft - 1);
          }
        }

        console.log('return this card', returnCardAtEndOfTurn);
      }

      if (drawDuringPostEffectPhase) {
        // TODO: ask to draw card or set activateAtEnd
        drawCard(3);
      }
    }
  }, [game.phase, isDiscardingCard, myPlayer.stable])


  // Effects for Effects =)
  useEffect(() => {
    if (drawFromOpponent) {
      console.log('Handle draw from opponent');
      // Select Opponent
      // Take card
      setSkipDrawPhase(true);
      setNumberOfEffectsHandled(numberOfEffectsHandled + 1);
    }
  }, [drawFromOpponent])

  useEffect(() => {
    if (borrowUnicorn) {
      console.log('Handle borrow unicorn');
      // Select unicorn from opponent stable
      setNumberOfEffectsHandled(numberOfEffectsHandled + 1);
    }
  }, [borrowUnicorn])


  // Effects for phases
  useEffect(() => {
    console.log(numberOfEffectsTotal, numberOfEffectsHandled)
    if (typeof numberOfEffectsTotal === 'number' && numberOfEffectsTotal === numberOfEffectsHandled) {
      console.log('ending pre effects phase')
      dispatch(nextPhase(1))
      socketServer.emit('endEffectPhase', lobbyName, 1);
    }
  }, [numberOfEffectsTotal, numberOfEffectsHandled])

  useEffect(() => {
    if (numberOfDrawsLeft === 0) {
      console.log('ending draw phase')
      dispatch(nextPhase(2));
    }
  }, [numberOfDrawsLeft])

  useEffect(() => {
    if (numberOfActionsLeft === 0) {
      console.log('ending action phase');
      if (myPlayer.hand.length > 7) {
        setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft + (myPlayer.hand.length - 7))
      }
      socketServer.emit('endActionPhase', lobbyName);
    }
  }, [numberOfActionsLeft])

  useEffect(() => {
    console.log('numberOfEndingEffectsLeft: ', numberOfEndingEffectsLeft)
    if (game.phase === 3 && numberOfEndingEffectsLeft === 0) {
      let nextTurn = game.turn + 1;
      let nextPlayerIndex = nextTurn % players.length;
      let whosTurn = players[nextPlayerIndex];

      const gameUpdates = {
        turn: nextTurn,
        whosTurn,
        phase: 0
      }

      dispatch(endTurn(gameUpdates, nextPlayerIndex === myPlayer.currentPlayerIndex));
      socketServer.emit('endTurn', lobbyName, gameUpdates, nextPlayerIndex);
    }
  }, [numberOfEndingEffectsLeft, game.phase])

  function addToStable() {
    const updatedPlayers = players;
    const updatedDecks = decks;
    const cardIndex = myPlayer.hand.findIndex(card => card.id === cardBeingPlayed.id);

    updatedPlayers[myPlayer.currentPlayerIndex].hand.splice(cardIndex,1);
    updatedPlayers[myPlayer.currentPlayerIndex].stable.push(cardBeingPlayed);

    return [updatedDecks, updatedPlayers];
  }

  function addToDiscardPile(shouldAlsoDiscardFromHand) {
    const updatedPlayers = players;
    const updatedDecks = decks;
    const opponentIndex = opponentInteractions.instant.playerIndex;
    const currentInstant = opponentInteractions.instant;
    const cardIndex = myPlayer.hand.findIndex(card => card.id === cardBeingPlayed.id);
    const instantIndex = updatedPlayers[opponentIndex].hand.findIndex(instant => instant.id === currentInstant.id);

    updatedDecks['discardPile'].push(cardBeingPlayed);
    updatedDecks['discardPile'].push(updatedPlayers[opponentIndex].instantIndex);
    updatedPlayers[myPlayer.currentPlayerIndex].hand.splice(cardIndex,1);
    updatedPlayers[opponentIndex].hand.splice(instantIndex,1);

    return [updatedDecks, updatedPlayers];
  }

  function handleEffects(effectCallback, requiresAction) {
    if (effectCallback) {
      effectCallback();
    }

    if (!requiresAction) {
      setNumberOfEffectsHandled(numberOfEffectsHandled + 1);
    }
  }

  function handleSideEffect(callback) {
    switch (cardBeingPlayed.upgrade) {
      case 12:
        //'See the top 3 cards on deck',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          show: decks.drawPile.slice(0, 3),
          callback
        })
        break;
      case 13:
        //you may search the deck for a "Bear Daddy Unicorn" card. Add it to your hand, then shuffle the deck.
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Search for bear daddy unicorn',
            callback: () => {
              const updatedPlayers = players;
              const updatedDecks = decks;
              const bearDaddyUnicornIndex = decks.drawPile.findIndex(card => card.name === 'BEAR DADDY UNICORN');

              if (bearDaddyUnicornIndex >= 0) {
                updatedPlayers[myPlayer.currentPlayerIndex].hand.push(updatedDecks.drawPile[bearDaddyUnicornIndex])
                updatedDecks.drawPile.splice(bearDaddyUnicornIndex, 1);
              }

              socketServer.emit('actionHappened', game.uri, Shuffle(updatedDecks), updatedPlayers);
              callback();
            }
          }]
        });
        break;
      case 14:
        //'When this card enters your Stable, you may choose a card from the discard pile and add it to your hand.',
        if (decks.discardPile.length === 0) {
          callback()
        }

        setActionSideEffect({
          name: cardBeingPlayed.name,
          modal: true,
          modalShow: decks.discardPile,
          modalAction: (card, index) => {
            const updatedDecks = decks;
            const updatedPlayers = players;

            updatedDecks.discardPile.splice(index, 1);
            updatedPlayers[myPlayer.currentPlayerIndex].hand.push(card);

            socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
            callback();
          },
          callback
        });
        break;
      case 18:
        //'When this card enters your Stable, you may search the deck for a "Twinkicorn" card. Add it to your hand, then shuffle the deck.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Search for Twinkicorn',
            callback: () => {
              const updatedPlayers = players;
              const updatedDecks = decks;
              const twinklecornIndex = decks.drawPile.findIndex(card => card.name === 'TWINKICORN');

              if (twinklecornIndex >= 0) {
                updatedPlayers[myPlayer.currentPlayerIndex].hand.push(updatedDecks.drawPile[twinklecornIndex])
                updatedDecks.drawPile.splice(twinklecornIndex, 1);
                updatedDecks.drawPile = Shuffle(updatedDecks.drawPile)
              }

              socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
              callback();
            }
          }]
        });
        break;
      case 19:
        //'When this card enters your Stable, you may choose any player. That player must DISCARD a card. If this card would be sacrificed or destroyed, return it to your hand instead.',
        //TODO: return if sacrified
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Make player discard card',
            callback: () => {
              dispatch(choosePlayer({
                isTrue: true,
                callback: (choosenPlayer) => {
                  setNumPlayersDiscarding(1);
                  setDiscardingAction({
                    callback
                  });

                  dispatch(choosePlayer({
                    isTrue: false,
                    callback: null
                  }))
                  socketServer.emit('playersDiscarding', game.uri, [choosenPlayer.index]);
                }
              }))
            }
          }]
        });
        break;
      case 21:
        //'When this card enters your Stable, you may DISCARD a card from the Nursery directly into your Stable.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Summon Baby Unicorn',
            callback: () => {
              summonBabyUnicorn();
              callback();
            }
          }]
        });
        break;
      case 22:
        //'When this card enters your Stable, you may return a card in any player\'s Stable to that player\'s hand.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Return Someone\'s Unicorn',
            callback: () => {
              dispatch(returningOpponentCard({
                isTrue: true,
                callback
              }))
            }
          }]
        });
        break;
      case 2:
        //'You may choose any other player. Pull a card from that player\'s hand and add it to your hand. If you do, skip your Draw phase.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          actions: [{
            name: 'Draw from Opponent',
            requiresAction: true,
            callback: () => dispatch(drawingFromOpponent({
              isTrue: true,
              callback
            }))
          }]
        });
        break;
      case 23:
        //'When this card enters your Stable, STEAL a Basic Unicorn card. If this card leaves your Stable, return that Basic Unicorn card to the Stable from which you stole it.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          actions: [{
            name: 'Borrow Unicorn',
            requiresAction: true,
            callback: () => dispatch(stealUnicorn({
              isTrue: true,
              type: 'Basic Unicorn',
              callback: callback
            }))
          }]
        });
        break;
      case 3:
        //'You may STEAL a Unicorn card. At the end of your turn, return that Unicorn card to the Stable from which you stole it.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          actions: [{
            name: 'Steal Basic Unicorn',
            requiresAction: true,
            callback: () => dispatch(drawingFromOpponent({
              isTrue: true,
              callback: (card, playerIndex) => handleEffects(() => {
                setReturnCardAtEndOfTurn({
                  ...card,
                  previousPlayerIndex: playerIndex,
                  currentPlayerIndex: myPlayer.currentPlayerIndex,
                  location: 'hand'
                });
                setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft + 1);
                callback();
              }, false)
            }))
          }]
        });
        break;
      case 25:
        //'When this card enters your Stable, choose a player and look at their hand. Choose a Unicorn card in that player\'s hand and move it to the discard pile.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Discard Someone\'s Unicorn',
            callback: () => {
              dispatch(discardingOpponentCard({
                isTrue: true,
                type: 'Unicorn',
                callback
              }))
            }
          }]
        });
        break;
      case 26:
        //'When this card enters your Stable, each player must SACRIFICE a Unicorn card.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Make everyone sacrifice card',
            callback: () => {
              setNumPlayersSacrifcing(players.length);
              setSacrficingAction({
                callback
              });
              let playerIndexes = [];
              for (var i = 0; i < players.length; i++) {
                playerIndexes.push(i);
              }
              socketServer.emit('playersSacrificing', game.uri, playerIndexes);
            }
          }]
        });
        break;
      case 27:
        // 'When this card enters your Stable, you may search the deck for a Unicorn card and add it to your hand. Shuffle the deck.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          modal: true,
          modalShow: decks.drawPile,
          modalAction: (card, index) => {
            if (card.type.match(/unicorn/ig)) {
              const updatedDecks = decks;
              const updatedPlayers = players;

              updatedDecks.drawPile.splice(index, 1);
              updatedPlayers[myPlayer.currentPlayerIndex].hand.push(card);
              updatedDecks.drawPile = Shuffle(updatedDecks.drawPile)

              socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
              callback();
            } else {
              //TODO add messaging
            }
          },
          callback
        });
        break;
      case 29:
        //'When this card enters your Stable, SACRIFICE all Downgrade cards.',
        const myDowngrades = groupBy(myPlayer.stable, 'type')['Downgrade']
        setActionSideEffect({
          name: 'Sacrifice Downgrades',
          show: myDowngrades,
          callback: () => {
            const updatedDecks = decks;
            const updatedPlayers = players;

            updatedDecks.discardPile.push(...myDowngrades);
            updatedPlayers[myPlayer.currentPlayerIndex].stable = myPlayer.stable.filter(card => card.type !== 'Downgrade');

            socketServer.emit('actionHappened', game.uri, Shuffle(updatedDecks), updatedPlayers);
            callback();
          }
        })
        break;
      case 31:
        //'When this card enters your Stable, STEAL a Baby Unicorn card. If this card leaves your Stable, return that Baby Unicorn card to the Stable from which you stole it.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          actions: [{
            name: 'Steal baby Unicorn',
            requiresAction: true,
            callback: () => dispatch(stealUnicorn({
              isTrue: true,
              type: 'Baby Unicorn',
              callback: callback
            }))
          }]
        });
        break;
      case 32:
        //'When this card enters your Stable, each other player must DISCARD a card.',
        setActionSideEffect({
          name: cardBeingPlayed.name,
          callback,
          actions: [{
            name: 'Make everyone discard card',
            callback: () => {
              setNumPlayersDiscarding(players.length - 1);
              setDiscardingAction({
                callback
              });
              let playerIndexes = [];
              for (var i = 0; i < players.length; i++) {
                if (i !== parseInt(myPlayer.currentPlayerIndex)) {
                  playerIndexes.push(i);
                }
              }
              socketServer.emit('playersDiscarding', game.uri, playerIndexes);
            }
          }]
        });
        break;
    }
  }

  function drawCard(phase) {
    let updatedDecks = decks;
    let player = myPlayer;
    let allPlayers = players;

    const nextCards = updatedDecks.drawPile.splice(0, 1);
    player.hand = [...player.hand, ...nextCards];
    allPlayers[player.currentPlayerIndex] = player;

    socketServer.emit('drawCard', lobbyName, updatedDecks, allPlayers, phase === 3 ? game.phase : phase)
    if (phase === 3) {
      if (game.phase === 2) {
        setNumberOfActionsLeft(numberOfActionsLeft - 1);
      } else {
        let endingEffectCountChange = 1;
        if (myPlayer.hand.length > 7) {
          endingEffectCountChange--;
        }
        setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft - endingEffectCountChange);
      }

    } else if (phase === 0) {
      // DO NOTHING
    } else {
      setNumberOfDrawsLeft(numberOfDrawsLeft - 1);
    }
  }

  function playCard() {
    dispatch(playingCard({isTrue: true, callback: null}))
  }

  function handleInstant(instant) {
    if (instant.name === 'Skip') {
      console.log('called skip intent')
      const [updatedDecks, updatedPlayers] = addToDiscardPile();
      socketServer.emit('endActionPhase', game.uri, 3, updatedDecks, updatedPlayers);
    } else {
      console.log('called play intent from player hand', instant)
      //TODO: adding ability to instant an instant
      // socketServer.emit('endActionPhase', game.uri, 3, updatedDecks, updatedPlayers);
    }
  }

  function handleActions(action, callback) {
    switch (action) {
      case 'drawCard':
        drawCard(3);
        break;
      case 'playCard':
        playCard();
        break;
      default:
        callback()
    }
  }

  function returnCardToOpponent(cardLocation, index, card) {
    let updatedPlayers = players;

    updatedPlayers[card.currentPlayerIndex][cardLocation].splice(index, 1);
    updatedPlayers[card.previousPlayerIndex][card.location].push(card);

    setReturnCardAtEndOfTurn(null);
    setNumberOfEndingEffectsLeft(numberOfEndingEffectsLeft - 1);
    socketServer.emit('returnCard', lobbyName, card, decks, updatedPlayers);
  }

  function summonBabyUnicorn() {
    let updatedPlayers = players;
    let updatedDecks = decks;
    let babyUnicorn = updatedDecks.nursery.pop()
    updatedPlayers[myPlayer.currentPlayerIndex].stable.push(babyUnicorn);

    socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
  }

  function sacrificeHand() {
    let updatedPlayers = players;
    let updatedDecks = decks;

    updatedDecks.discardPile = [...updatedDecks.discardPile, ...myPlayer.hand]
    updatedPlayers[myPlayer.currentPlayerIndex].hand = [];

    socketServer.emit('actionHappened', game.uri, updatedDecks, updatedPlayers);
  }

  function renderView() {
    switch (game.phases[game.phase].name) {
      case 'Pre Effects':
        return <EffectsView
                  effects={effects}
                  handleEffects={handleEffects}
                  numberOfEffectsHandled={numberOfEffectsHandled}
                  allowSkip={allowSkip}
                  skipPhase={() => setNumberOfEffectsHandled(numberOfEffectsHandled + 1)}/>
        break;
      case 'Draw':
        return (
          <DrawView
            drawCard={drawCard}
          />
        )
        break;
      case 'Action':
        return <ActionView
                  handleActions={handleActions}
                  checkForInstant={checkForInstant}
                  handleInstant={handleInstant}
                  actionSideEffect={actionSideEffect}/>
        break;
      case 'Post Effects':
        return <EndView />
        break;
    }
  }

  function renderPlayersToAttack() {
    if (isDestroyingCard.isTrue ||
      isDrawingFromOpponent.isTrue ||
      isChoosingPlayer.isTrue ||
      isStealingUnicorn.isTrue ||
      isReturningOpponentCard.isTrue ||
      isDiscardingOpponentCard.isTrue
    ) {
      return <PlayersToAttackComponent

      />
    }
  }

  return (
    <div>
      My Turn
      {renderView()}

      { renderPlayersToAttack() }
    </div>
  )
}

function EffectsView(props) {
  const { handleEffects, allowSkip, skipPhase, effects, numberOfEffectsHandled } = props;

  function renderEffects() {
    if (effects[numberOfEffectsHandled]) {
      return (<div>{
        effects[numberOfEffectsHandled].actions.map((action, index) => {
          return <Button key={index} onClick={() => { handleEffects(action.callback, action.requiresAction) }}>{action.name}</Button>
        })
      }</div>)
    }
  }


  return (
    <div>
      { renderEffects() }
      { allowSkip ? <Button onClick={skipPhase}>Skip</Button> : null }
    </div>
  )
}

function DrawView(props) {
  const { drawCard } = props;
  return (
    <div>
      <Button onClick={() => { drawCard(2) }}>Draw Card</Button>
    </div>
  )
}

// TODO: make shared component with spectatorview
function ActionView(props) {
  const { handleActions, checkForInstant, handleInstant, actionSideEffect } = props;
  const instantActions = useCheckForInstants;
  const isDestroyingCard = useSelector(state => state.isDestroyingCard);
  console.log('Should be showing instants: ', checkForInstant)

  function renderInstants() {
    if (checkForInstant) {
      return (
        <div>
          { instantActions.map(action => {
            return (
              <Button key={action.id}
                content={action.name}
                onClick={() => handleInstant(action) }
              />
            )
          })}
        </div>
      )
    }
  }

  function renderActionSideEffects() {
    if (actionSideEffect && actionSideEffect.show) {
      return (
        <div>
          <Card.Group>
            {
              actionSideEffect.show.map((card, index) => {
                return <CardComponent
                  index={index}
                  key={card.id}
                  card={card}
                  callback={() => {}}
                />
              })
            }
          </Card.Group>

          <Button onClick={actionSideEffect.callback}>End Action Phase</Button>
        </div>
      )
    }

    if (actionSideEffect && actionSideEffect.actions) {
      return (
        <div>
          {
            actionSideEffect.actions.map((action, index) => {
              return <Button onClick={action.callback}>{action.name}</Button>
            })
          }

          <Button onClick={actionSideEffect.callback}>End Action Phase</Button>
        </div>
      )
    }

    if (actionSideEffect && actionSideEffect.modal) {
      return <ModalComponent
        header="Select card to draw"
        cards={actionSideEffect.modalShow}
        close={() => {}}
        callback={actionSideEffect.modalAction}/>
    }
  }

  return (
    <div>
      <Button onClick={() => { handleActions('drawCard') }}>Draw Card</Button>
      <Button onClick={() => { handleActions('playCard') }}>Play Card</Button>
      { renderActionSideEffects() }
      { renderInstants() }
    </div>
  )
}

function PlayersToAttackComponent(props) {
  const {} = props;
  const myPlayerIndex = useSelector(state => state.currentPlayerIndex);
  const lobbyName = useSelector(state =>  state.game.uri);
  const socketServer = useSelector(state => state.socket);
  const players = useSelector(state =>  state.players);
  const decks = useSelector(state =>  state.decks);
  const isDestroyingCard = useSelector(state => state.isDestroyingCard);
  const isDrawingFromOpponent = useSelector(state => state.isDrawingFromOpponent);
  const isGivingToOpponent = useSelector(state =>  state.isGivingToOpponent);
  const isChoosingPlayer = useSelector(state =>  state.isChoosingPlayer);
  const isStealingUnicorn = useSelector(state =>  state.isStealingUnicorn);
  const isReturningOpponentCard = useSelector(state =>  state.isReturningOpponentCard);
  const isDiscardingOpponentCard = useSelector(state => state.isDiscardingOpponentCard);
  const dispatch = useDispatch();

  const [selectedPlayer, setSelectedPlayer] = useState(null);

  useEffect(() => {
    if (selectedPlayer && isChoosingPlayer.isTrue) {
      setSelectedPlayer(null);
      if (isChoosingPlayer.card) {
        handleGiveToOpponent(selectedPlayer);
      } else {
        isChoosingPlayer.callback(selectedPlayer)
      }
    }
  }, [selectedPlayer])

  function renderPlayers() {
    if (!selectedPlayer) {
      return (
        <div>
        {
          players.map((player, index) => {
            if (index !== parseInt(myPlayerIndex)) {
              return (
                <Card
                  raised
                  key={player.id}
                  onClick={() => { setSelectedPlayer({...player, index}) }}>
                  <Image
                  label={{
                      color: player.color,
                      content: `${player.name}: H: ${player.hand.length} S: ${player.stable.length}`,
                      ribbon: true
                    }}
                   src={player.unicorn.url}/>
                </Card>
              )
            }
          })
        }
        </div>
      )
    }
  }

  function renderPlayerCards() {
    let handleAction;

    if (isDestroyingCard.isTrue) {
      handleAction = handleDestroyCard;
    }

    if (isDrawingFromOpponent.isTrue) {
      handleAction = handleDrawFromOpponent;
    }

    if (isStealingUnicorn.isTrue) {
      handleAction = (card, index) => {
        if (isChoosingPlayer.isTrue) {
          handleGiveToOpponent(selectedPlayer);
        }
        handleStealUnicorn(card, index);
      }
    }

    if (isReturningOpponentCard.isTrue) {
      handleAction = handleReturnOpponentCard;
    }

    if (isDiscardingOpponentCard.isTrue) {
      handleAction = handleDiscardingOpponentCard;
    }

    if (selectedPlayer) {
      return <Card.Group>
          {
            selectedPlayer[isDestroyingCard.isTrue || isStealingUnicorn.isTrue || isReturningOpponentCard.isTrue ? 'stable' : 'hand'].map((card, index) => {
              return <CardComponent
                index={index}
                key={card.id}
                card={card}
                callback={handleAction}
              />
            })
          }
        </Card.Group>
    }
  }

  function handleDestroyCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    if (card.type === 'Baby Unicorn') {
      updatedDecks['nursery'].push(card);
    } else {
      updatedDecks['discardPile'].push(card);
    }

    updatedPlayers[selectedPlayer.index].stable.splice(index,1);

    socketServer.emit('destroyCard', lobbyName, card, updatedDecks, updatedPlayers);
    if (isDestroyingCard.callback) {
      isDestroyingCard.callback();
    }

    dispatch(destroyingCard({isTrue: false, callback: null}))
  }

  function handleDrawFromOpponent(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    updatedPlayers[myPlayerIndex].hand.push(card);
    updatedPlayers[selectedPlayer.index].hand.splice(index,1);

    socketServer.emit('drawFromOpponent', lobbyName, card, updatedDecks, updatedPlayers);
    if (isDrawingFromOpponent.callback) {
      isDrawingFromOpponent.callback(card, selectedPlayer.index);
    }

    dispatch(drawingFromOpponent({isTrue: false, callback: null}))
  }

  function handleStealUnicorn(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    if (!isStealingUnicorn.type || isStealingUnicorn.type === card.type) {
      updatedPlayers[selectedPlayer.index].stable.splice(index, 1);
      updatedPlayers[myPlayerIndex].stable.push(card);
    }

    socketServer.emit('stealUnicorn', lobbyName, card, updatedDecks, updatedPlayers);
    if (isStealingUnicorn.callback) {
      isStealingUnicorn.callback();
    }

    dispatch(stealUnicorn({isTrue: false, type: null, callback: null}))
  }

  function handleGiveToOpponent(selectedPlayer) {
    const card = isChoosingPlayer.card;
    const updatedDecks = decks;
    const updatedPlayers = players;

    updatedPlayers[myPlayerIndex].stable.splice(card.cardIndex, 1);
    updatedPlayers[selectedPlayer.index].stable.push(card);

    socketServer.emit('giveToOpponent', lobbyName, card, updatedDecks, updatedPlayers);
    if (isChoosingPlayer.callback) {
      isChoosingPlayer.callback(selectedPlayer.index);
    }

    dispatch(choosePlayer({isTrue: false, card:null, callback: null}))
  }

  function handleReturnOpponentCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    updatedPlayers[selectedPlayer.index].stable.splice(index, 1);
    updatedPlayers[selectedPlayer.index].hand.push(card);

    socketServer.emit('returningOpponentCard', lobbyName, card, updatedDecks, updatedPlayers);
    if (isReturningOpponentCard.callback) {
      isReturningOpponentCard.callback();
    }

    dispatch(returningOpponentCard({isTrue: false, callback: null}))
  }

  function handleDiscardingOpponentCard(card, index) {
    const updatedDecks = decks;
    const updatedPlayers = players;

    updatedPlayers[selectedPlayer.index].hand.splice(index, 1);
    updatedDecks['discardPile'].push(card);

    socketServer.emit('discardingOpponentCard', lobbyName, card, updatedDecks, updatedPlayers);
    if (isDiscardingOpponentCard.callback) {
      isDiscardingOpponentCard.callback();
    }

    dispatch(discardingOpponentCard({isTrue: false, type: null, callback: null}))
  }

  return (
    <div>
      <Header>{selectedPlayer ? "Select a Card" : "Choose A Player"}</Header>

      { renderPlayers() }
      { renderPlayerCards() }
    </div>
  )
}

function EndView(props) {
  const {} = props;
  return (
    <div>
      Ending turn
    </div>
  )
}

export default PlayerView
