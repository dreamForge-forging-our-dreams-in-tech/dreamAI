import fs from 'fs';
import path from 'path';

// Read the file as a string
const rawData = fs.readFileSync(path.join(process.cwd(), 'AI/tokenizer/tokenizer_data.json'), 'utf8');

// Convert string to a JS Object
const tokenizer_data = JSON.parse(rawData);

let increase_id = 1; // by how much every new id is increased

//class for handling tokenization functions such as tokinization, de tokizination, training etc
class Tokenizer {

    constructor() { }

    tokenize(input) { // tokenize a string input
        if (!input) return "";

        let processedInput = input.trim().toLowerCase();

        // Sort keys by length (longest first) so "apple" matches before "app"
        const sortedKeys = Object.keys(tokenizer_data).sort((a, b) => b.length - a.length);

        //add a new word that it doesn't know yet to tokenizer_data.json
        let new_sentence = [];
        processedInput.split(' ').forEach(word => {
            if (!sortedKeys.includes(word)) {
                this.add_word(word);
            }
            new_sentence.push(tokenizer_data[word])
        });


        //console.log("Tokenized Input:", processedInput);
        return new_sentence;
    }

    de_tokenize(input) { // turn a tokenized string of numbers back into words
        if (!input) return "";

        // 1. Invert the tokenizer_data (Map IDs to Words)
        const reverseData = {};
        Object.entries(tokenizer_data).forEach(([word, id]) => {
            reverseData[id] = word;
        });

        let processedInput = input.toString().trim();

        // 2. Sort ID keys by length (longest first) 
        // This prevents "11" being caught by the rule for "1"
        const sortedIds = Object.keys(reverseData).sort((a, b) => b.length - a.length);

        // 3. Replace IDs with Words
        sortedIds.forEach(id => {
            const escapedId = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

            // Use a Word Boundary (\b) if your tokens are space-separated 
            // to avoid accidental matches inside larger numbers
            const regex = new RegExp(escapedId, 'g');
            processedInput = processedInput.replace(regex, reverseData[id]);
        });

        return processedInput;
    }

    add_word(word) { // function to add any unknown word to the tokenizer_data.json
        let IDs = Object.values(tokenizer_data);

        if (IDs.length === 0) IDs.push(0)

        let new_id = IDs[IDs.length - 1] + increase_id; // +1 because it counts from 1 and not from 0
        
        if (IDs.includes(new_id)) {
            new_id = new_id + increase_id; //if retireving the length fails or there is a duplicate grab the last id and increase it.
        }

        tokenizer_data[word] = new_id;

        const jsonData = JSON.stringify(tokenizer_data, null, 2);

        try {
            fs.writeFileSync('AI/tokenizer/tokenizer_data.json', jsonData);
            console.log("File saved successfully!");
        } catch (err) {
            console.error("Error writing file:", err);
        }
    }

}

export { Tokenizer };