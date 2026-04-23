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

    // function that "prepares" the training data for tokenization.
    // it finds the most used words from the training_array and lists them from msot used and least used.
    // and then adds those words to the tokenizer_data.json with an id that is increased by 1 for every new word.
    // it returns the result.
    prepare_data(training_array) {
        let i;
        let words_counter = {};

        for (i of training_array) {
            i.split(' ').forEach(word => {
                words_counter[word] = (words_counter[word] || 0) + 1; // Count the frequency of each word
                words_counter = Object.fromEntries(Object.entries(words_counter).sort((a, b) => b[1] - a[1])); // Sort by frequency (most used first)

               for (let word in words_counter) {
                word = word.trim().toLowerCase();
                    if (!tokenizer_data.hasOwnProperty(word)) {
                        this.add_word(word);
                    }
                }
            });
        }

        return words_counter;
    }

    tokenize(input) { // tokenize a string input
        if (!input) return "";

        let processedInput = input.trim().toLowerCase();

        // Sort keys by length (longest first) so "apple" matches before "app"
        const sortedKeys = Object.keys(tokenizer_data).sort((a, b) => b.length - a.length);

        //add a new word that it doesn't know yet to tokenizer_data.json
        let new_sentence = [];
        processedInput.split(' ').forEach(word => {
            if (!tokenizer_data.hasOwnProperty(word)) {
                this.add_word(word);
            }
            new_sentence.push(tokenizer_data[word]);
        });

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

        if (IDs.length === 0) {
            IDs.push(0); // Start from 0 if there are no existing IDs
            tokenizer_data['<end>'] = 0; // make the end token always be 0 and the first token in the tokenizer to evoid it from crashing if its empty.
        }

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