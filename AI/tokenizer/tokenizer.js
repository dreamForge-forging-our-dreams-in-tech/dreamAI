import fs from 'fs';
import path from 'path';

// Read the file as a string
const rawData = fs.readFileSync(path.join(process.cwd(), 'AI/tokenizer/tokenizer_data.json'), 'utf8');

// Convert string to a JS Object
const tokenizer_data = JSON.parse(rawData);

function tokenize(input) {
    if (!input) return "";
    
    let processedInput = input.trim().toLowerCase();
    
    // Sort keys by length (longest first) so "apple" matches before "app"
    const sortedKeys = Object.keys(tokenizer_data).sort((a, b) => b.length - a.length);

    sortedKeys.forEach(key => {
        // This line escapes special regex characters like ?, ., *, +, etc.
        const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        
        // Use the escaped key in the RegExp
        const regex = new RegExp(escapedKey.toLowerCase(), 'g');
        processedInput = processedInput.replace(regex, tokenizer_data[key]);
    });
    
    console.log("Tokenized Input:", processedInput);
    return processedInput;
}

export { tokenize };