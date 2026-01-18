// Handle image preview
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');

imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
});

// Handle form submission
const form = document.getElementById('analysisForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = submitBtn.querySelector('.btn-text');
const loader = submitBtn.querySelector('.loader');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

form.addEventListener('submit', async function(e) {
    e.preventDefault();

    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Processing...';
    loader.style.display = 'inline-block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    // Prepare form data
    const formData = new FormData(form);

    try {
        // Send request
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        // Display error
        displayError(error.message, error.trace || '');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        btnText.textContent = 'Analyze Image';
        loader.style.display = 'none';
    }
});

function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    errorSection.style.display = 'none';

    // Step 6: Final Answer (most important)
    document.getElementById('finalAnswer').textContent = data.step6.answer;

    // Token Comparison
    document.getElementById('baselineTokens').textContent =
        `${data.comparison.baseline_tokens} tokens`;
    document.getElementById('ourTokens').textContent =
        `${data.comparison.our_tokens} tokens`;
    document.getElementById('savingsPercent').textContent =
        `${data.comparison.savings_percent.toFixed(1)}% saved`;

    // Step 2: Prompt Compression
    document.getElementById('step2Original').textContent = data.step2.original_prompt;
    document.getElementById('step2Compressed').textContent = data.step2.compressed_prompt;
    document.getElementById('step2Tokens').textContent =
        `${data.step2.original_tokens} → ${data.step2.compressed_tokens} tokens (${data.step2.savings_percent.toFixed(1)}% saved)`;

    // Step 3: CVspec
    document.getElementById('step3Modules').textContent =
        data.step3.active_modules.length > 0
            ? data.step3.active_modules.join(', ')
            : 'none';
    document.getElementById('step3Cvspec').textContent =
        JSON.stringify(data.step3.cvspec, null, 2);

    // Step 4: Vision Pipeline
    document.getElementById('step4Output').textContent = data.step4.raw_output;
    document.getElementById('step4Tokens').textContent = `${data.step4.output_tokens} tokens`;

    // Step 5: Compression
    document.getElementById('step5ImageDesc').textContent = data.step5.compressed_image_desc;
    document.getElementById('step5Question').textContent = data.step5.compressed_question;
    document.getElementById('step5Tokens').textContent =
        `${data.step5.total_compressed} tokens (${data.step5.savings_percent.toFixed(1)}% saved)`;

    // Step 6: Final Prompt
    document.getElementById('step6Prompt').textContent = data.step6.final_prompt;
    document.getElementById('step6Tokens').textContent = `${data.step6.final_prompt_tokens} tokens`;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayError(message, trace) {
    resultsSection.style.display = 'block';
    errorSection.style.display = 'block';

    document.getElementById('errorMessage').textContent = message;
    if (trace) {
        document.getElementById('errorTrace').textContent = trace;
    }

    // Scroll to error
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
