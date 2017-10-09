/**
 * Neuron-Driven-Development
 *
 */
(function () {
    window.ndd = ndd

    // returns a neural network that implements the map provided
    // map is a string representing the neurons and synapses
    function ndd(map) {
        var neurons = Array.from(map).map(function () {
            return { state: false, layer: 0 }
        })

        neurons.render = render.bind(neurons)
        neurons.sendInt = sendInt.bind(neurons)
        neurons.sendText = sendText.bind(neurons)

        return neurons
    }

    // send an ascii-encoded text value as input
    function sendText(v) {
        return this.sendInt(v.charCodeAt(0))
    }

    function sendInt(v) {
        var neurons = this
        for (var i = 7; i >= 0; i--) {
            neurons[i].state = v & 1
            v >>= 1
        }
    }

    function render() {
        var neurons = this

        var ndd = document.createElement('div')
        var style = document.createElement('style')
        style.textContent = STYLE
        ndd.appendChild(style)

        // create a domElement for each neuron
        for (var n of neurons) {
            n.domElement = document.createElement('div')
            n.domElement.className = 'neuron'
            n.domElement.addEventListener('click', function (n) {
                n.state = !n.state
            }.bind(null, n))
        }

        // group neurons by layers
        var layers = []
        for (var n of neurons) {
            if (!layers[n.layer]) {
                layers[n.layer] = []
                layers[n.layer].domElement = document.createElement('div')
                layers[n.layer].domElement.className = 'layer'
                ndd.appendChild(layers[n.layer].domElement)
            }

            layers[n.layer].push(n)
            layers[n.layer].domElement.appendChild(n.domElement)
        }

        function update() {
            for (var n of neurons) {
                n.domElement.classList.toggle('active', n.state)
            }
            requestAnimationFrame(update)
        }

        update()

        return ndd
    }


    var STYLE = `
    .neuron {
        width: 20px;
        min-width: 20px;
        max-width: 20px;
        height: 20px;
        border: 1px solid black;
        border-radius: 50%;
        margin: 10px;
    }

    .neuron.active {
        background: green;
    }

    .layer {
        display: flex;
        /*width: 500px;*/
        /*margin: 0 auto;*/
    }`
})()
