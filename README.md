<div class="markdown prose w-full break-words dark:prose-invert light">
    <h1>README for Value Class</h1>
    <h2>Overview</h2>
    <p>The Value class is an implementation of a computational graph node that supports basic mathematical operations
        such as addition, multiplication, division, subtraction, negation, etc. and implements forward-mode automatic
        differentiation. The forward-mode automatic differentiation allows the computation of gradients of mathematical
        functions defined as a computation graph.</p>
    <h2>Features</h2>
    <ul>
        <li>Supports basic mathematical operations such as addition, subtraction, multiplication, division, negation,
            and power.</li>
        <li>Implements forward-mode automatic differentiation and allows the computation of gradients.</li>
        <li>Supports ReLU activation function.</li>
    </ul>
    <h2>Usage</h2>
    <h3>Creating an instance</h3>
    <p>Create a new instance of the Value class by passing the initial data to it. For example:</p>
    <pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class=""></span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">v = Value(<span class="hljs-number">3</span>)
</code></div></div></pre>
    <h3>Performing mathematical operations</h3>
    <p>Perform mathematical operations on the nodes of the computational graph using the basic mathematical operations
        as methods. For example:</p>
    <pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class=""></span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">w = v + <span class="hljs-number">5</span>
x = v * <span class="hljs-number">2</span>
y = x ** <span class="hljs-number">3</span>
</code></div></div></pre>
    <h3>Computing gradients</h3>
    <p>Compute the gradients of the computational graph with respect to a particular node by calling the
        <code>backward</code> method on that node. For example:</p>
    <pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class=""></span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">y.backward()
</code></div></div></pre>
    <p>The gradients with respect to each node can be accessed using the <code>grad</code> attribute of each node. For
        example:</p>
    <pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><span class=""></span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python"><span class="hljs-built_in">print</span>(v.grad)
</code></div></div></pre>
    <h2>Note</h2>
    <ul>
        <li>The order of operations is determined using the directed acyclic graph of the computational graph.</li>
        <li>The forward-mode automatic differentiation only works with scalar-valued functions.</li>
    </ul>
</div>
