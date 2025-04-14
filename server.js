// server.js
const express = require('express');
const app = express();
const PORT = 3001;

app.use(express.json());
app.use(express.static('public')); // Your HTML file should be inside /public folder

app.post('/chat', (req, res) => {
    const message = req.body.message;
    res.json({ response: `You said: ${message}. Here's some advice.` });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
