# dreamAI

dreamAi is dreamForge its attempt at creating an advanced roleplaying ai that can run on hardware from the 2000s.

# Licensing
You can find the license for this project under LICENSE.
Please be aware that when using this for any comemrcially available product you are required to provide credits towards dreamForge.

# Tested devices

* Modified ASRock Vision 100ht from 2010.
* Acer Aspire 5 (tested before transfering to ASRock)

# How it works?

dreamAi uses tensorflow.tfjs to build its llm and uses character cards and controlled training data, context windows, variables and some other stuff to ensure the AI doesnt make your 200s pc catch fire because its writting is like shakespeare.
<https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT8GWQKXDIsvdo2Crjeaq2R4kU8WZKZGJa3xA&s>

# API
Although dreamAi comes with its own frontend users are allowed to build their own front ends and can use the below API endpoints to do so!

## /chat

## /memory
This is a GET api endpoint and returns the total memory used by the AI and the node process.

## /progress
This is a GET api endpoint that returns a JSON with information regarding the training progress of the AI.

{
    epoch: epoch + 1, -- the current epoch it's training
    loss: logs.loss.toFixed(4), -- the loss of the epoch
    total_epochs: this.params.epochs -- total epochs it going to run.
}
