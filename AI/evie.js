import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';

class EvieOptimizer extends tf.Optimizer {
    constructor(learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.accumulators = new Map();

        this.weightMap = new Map(); // 🔒 Private internal map
    }

    // Define the actual math step
    applyGradients(variableGradients) {
        this.t++;

        let registered_variables = tf.engine().registeredVariables;

        tf.tidy(() => {
            const biasCorrection1 = 1 - Math.pow(this.beta1, this.t);
            const biasCorrection2 = 1 - Math.pow(this.beta2, this.t);

            const nextM_math = 1 - this.beta1;
            const nextV_math = 1 - this.beta2;

            for (const gradientObj of variableGradients) {

                // From your JSON: { name: 'dense_Dense1/kernel', tensor: Tensor }
                const varName = gradientObj.name;
                const grad = gradientObj.tensor;

                if (!grad) { // skip tensors with null
                    continue;
                }

                const variable = registered_variables[varName];

                if (!this.accumulators.has(varName)) {
                    this.accumulators.set(varName, {
                        m: tf.zerosLike(variable).variable(),
                        v: tf.zerosLike(variable).variable()
                    });
                }

                const { m, v } = this.accumulators.get(varName);

                const nextM = m.add(grad.sub(m).mul(nextM_math));
                const nextV = v.add(grad.square().sub(v).mul(nextV_math));

                const mHat = nextM.div(biasCorrection1);
                const vHat = nextV.div(biasCorrection2);

                const update = mHat.mul(this.learningRate).div(vHat.add(this.epsilon).sqrt());

                variable.assign(variable.sub(update));

                m.assign(nextM);
                v.assign(nextV);
            }
        });
    }

    static get className() {
        return 'EvieOptimizer';
    }
}

// 💥 CRITICAL STEP: Registers it with TensorFlow.js
tf.serialization.registerClass(EvieOptimizer);

// Helper function to match your exact syntax: evie(0.01)
function Evie(lr) {
    return new EvieOptimizer(lr);
}

export { Evie }