# Test1

Video: https://youtu.be/yWGm-uIacnM

Total training time: 2:40
Max temp: 38 degrees
Used ram: ~ 7

This was a starting test to collect starting data for upcomming tests.

# test 2 with Evie

total training time: 2:20

max temp: 41 degrees
used ram: ~6

Keeps stuck at a loss 0f 3.3673

changed learning rate from 0.01 to 0.001 or 0.1 > got incredibly slow at learning and had fluctuating results.

Okay turns out grad is always undefined in Evie.js which causes the empty tensor detecting to skip the loop.

Okay turns out model.fit was returning an array and not a object, took some time to figurie it out with gemini.
, if gemini just fucking told me immediatly i needed to acces tensor in the object instead of a fucking number.

# test 3 evie looping array and properly accesing objects and tensors

total training time: 2:45
max temp: 38 degrees.
used ram: ~6

Reposne: similar results as to adam.

# teest 4 evie with batching

Slower, training takes longer and loss has become worse than before
So back to the drawing board.

# test 5 with tidy and removed dataSync

Slightly improved training speeds it got under 0.5 after 22 (25 previously) epochs in 1:50 minutes
Above results do seem to vary

# test 6 imrpoved math, shorter is faster

this:
                const nextM = m.mul(this.beta1).add(grad.mul(1 - this.beta1));
                const nextV = v.mul(this.beta2).add(grad.square().mul(1 - this.beta2));
Became:
                const nextM = m.add(grad.sub(m).mul(1 - this.beta1));
                const nextV = v.add(grad.square().sub(v).mul(1 - this.beta2));

Minor improvements :(

# test 7

this: const update = mHat.mul(this.learningRate).div(vHat.sqrt().add(this.epsilon));
to: const update = mHat.mul(this.learningRate).div(vHat.add(this.epsilon).sqrt());

This provided a small but major improvement, it now would reach under 0.5 before 25 instead of around 25(23 - 27) like previous improvements did.
Because of this it seems responses also improved but due to limited training data this is up to doubt

Test 7 - pre computing

From:

                const nextM = m.add(grad.sub(m).mul(1 - this.beta1));
                const nextV = v.add(grad.square().sub(v).mul(1 - this.beta2));

                const mHat = nextM.div(1 - Math.pow(this.beta1, this.t));
                const vHat = nextV.div(1 - Math.pow(this.beta2, this.t));
To:
            const biasCorrection1 = 1 - Math.pow(this.beta1, this.t);
            const biasCorrection2 = 1 - Math.pow(this.beta2, this.t);

            const nextM_math = 1 - this.beta1;
            const nextV_math = 1 - this.beta2;
            
                const nextM = m.add(grad.sub(m).mul(nextM_math));
                const nextV = v.add(grad.square().sub(v).mul(nextV_mathz));

                const mHat = nextM.div(biasCorrection1);
                const vHat = nextV.div(biasCorrection2);

This added minor improvements, its loss still gets under 0.5 around 22 - 24 epochs but the number is closer/lower than previously (e.g. 0.499 to 0.480)
Sometimes it hits 0.3 now before 22.

# test 8 - video

video: https://youtu.be/yiyyAPtIuXk

Training time: 2:30
max temp: 37 degrees
used ram: 6.30

-- added scrolling to bottom if the users messages overflows.

-- shortened math formulas for increased speed, although it did not give much improvement.

-- added pre computing to Evie its for loop.