// optimiser_worker.cjs
module.exports = (data) => {
    const { range, buffers, params } = data;
    const [start, end] = range;

    // View the shared memory
    const grad = new Float32Array(buffers.sharedGrad);
    const m = new Float32Array(buffers.sharedM);
    const v = new Float32Array(buffers.sharedV);
    const varArr = new Float32Array(buffers.sharedVar);

    for (let i = start; i < end; i++) {
        // Adam Update Logic
        m[i] = m[i] + (grad[i] - m[i]) * params.nextM_math;
        v[i] = v[i] + (Math.pow(grad[i], 2) - v[i]) * params.nextV_math;

        const update = (m[i] * params.lr) / (Math.sqrt(v[i]) + params.eps);
        varArr[i] -= update;
    }

    return 'done';
};