// PeerJS signaling broker for Least Count.
// One small Node process: handles peer registration + connection setup, then steps
// aside while the actual game traffic flows P2P browser-to-browser.
const { PeerServer } = require('peer');

const PORT = Number(process.env.PORT) || 9000;
const PATH = process.env.PEER_PATH || '/lc';

const server = PeerServer({
  port: PORT,
  path: PATH,
  allow_discovery: false,
  proxied: true,
});

server.on('connection', (client) => {
  console.log(`[+] ${client.getId()} (${server._clients?.length ?? '?'} connected)`);
});
server.on('disconnect', (client) => {
  console.log(`[-] ${client.getId()}`);
});

console.log(`PeerJS broker listening on port ${PORT}, path ${PATH}`);
