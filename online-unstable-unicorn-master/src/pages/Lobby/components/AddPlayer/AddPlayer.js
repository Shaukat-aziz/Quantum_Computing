import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useMyPlayer } from 'utils/hooks.js';
import GroupBy from 'lodash/groupBy';
import Reduce  from 'lodash/reduce';
import { useParams } from "react-router-dom";
import { Header, Image, Icon, Segment, Button, Input, Label } from 'semantic-ui-react';

const colors = ['purple', 'blue', 'teal', 'green', 'yellow', 'orange', 'red'];

function AddPlayer(props) {
    const myPlayer = useMyPlayer();
    const numPlayers = useSelector(state => state.players.length);
    const socketServer = useSelector(state => state.socket);
    const [shouldOpenModal, setShouldOpenModal] = useState(false);
    const [username, setUsername] = useState("");
    const [unicorn, setUnicorn] = useState({});
    const urlParams = useParams().id;
    const dispatch = useDispatch();

    useEffect(() => {
        // localStorage.setItem('currentPlayerIndex', '')
    }, [])

    function toggleAvatarSelector() {
        setShouldOpenModal(!shouldOpenModal)
    }

    function renderForm() {
        if (!myPlayer.currentPlayerIndex) {
            return <MemoAddPlayerForm
                unicorn={unicorn}
                username={username}
                toggleAvatarSelector={toggleAvatarSelector}
                addPlayer={addPlayer}
                setUsername={setUsername}
            />
        }
    }

    function renderModal() {
        if (shouldOpenModal) {
            return <AvatarModal
                unicorn={unicorn}
                callback={handleSelectUnicorn}
                toggleAvatarSelector={toggleAvatarSelector}
            />
        }
    }

    function handleSelectUnicorn(unicorn, index) {
        //TODO: update players so 2 players can't set the same unicorn
        setUnicorn({
            index,
            ...unicorn
        });
        toggleAvatarSelector()
    }


    function addPlayer() {
        const newPlayer = {
          id: numPlayers + 1,
          connected: true,
          color: colors[numPlayers + 1],
          name: username,
          unicorn,
          hand: [],
          stable: [unicorn],
          upgrades: [],
          downgrades: [],
          playingCard: false
        };

        localStorage.setItem('currentPlayerIndex', numPlayers);
        socketServer.emit('addPlayer', urlParams, newPlayer);
    }

    return (
        <div>
            {renderForm()}
            {renderModal()}
        </div>
    )
}

const MemoAddPlayerForm = React.memo((props) => {
    const { unicorn, username, toggleAvatarSelector, setUsername, addPlayer} = props;

    return (
        <div>
            <Button
            content="Select Unicorn"
            onClick={toggleAvatarSelector}
            className='icon'
            icon='plus'
            />

            { unicorn.id ? <Image src={unicorn.url} avatar /> : null }
            Name: <Input value={username} id="addUserText" onChange={(e) => setUsername(e.target.value)} /> <Button onClick={addPlayer}>Add</Button>
        </div>
    )
})

function AvatarModal(props) {
    const { unicorn, callback, toggleAvatarSelector } = props;
    const cards = useSelector(state => state.game.cards);
    const socketServer = useSelector(state => state.socket);
    const players = useSelector(state => state.players);
    const [babyUnicorns, setBabyUnicorns] = useState(GroupBy(cards, 'type')['Baby Unicorn']);

    useEffect(() => {

    }, [socketServer])

    useEffect(() => {
      //TODO: update to use filter instead
        const usedUnicorns = Reduce(players, (newArr, player) => {
            return [...newArr, player.unicorn.id];
        }, []);

        const babyUnicornsRemaining = []

        for (var i = 0; i < babyUnicorns.length; i++) {
          let unicornIndex = usedUnicorns.findIndex(usedUnicornId => {
              return usedUnicornId === babyUnicorns[i].id
          })

          if (unicornIndex === -1) {
              babyUnicornsRemaining.push(babyUnicorns[i])
          }
        }

        setBabyUnicorns(babyUnicornsRemaining);
    }, [players, unicorn])

    return (
        <Segment id="unicorn-avatars">
            <Header>Available Unicorns</Header>
            <Label icon="close" corner="right" color="red" onClick={toggleAvatarSelector} />
            {babyUnicorns.map((option, i) => (
                <Image
                    className={`unicorn-avatar ${unicorn.id === option.id ? 'selected' : ''}`}
                    key={option.id}
                    text={option.name}
                    src={option.url}
                    size="small"
                    circular
                    onClick={() => { callback(option, i)}}  />
            ))}
        </Segment>
    )
}

export default AddPlayer;
