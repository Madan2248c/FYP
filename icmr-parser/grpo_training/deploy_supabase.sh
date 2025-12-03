#!/bin/bash

# Deployment script for Supabase Edge Function
# This script automates the deployment of the evaluation API

set -e  # Exit on error

echo "=========================================="
echo "Supabase Edge Function Deployment"
echo "=========================================="
echo ""

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI not found!"
    echo ""
    echo "Install it with:"
    echo "  macOS: brew install supabase/tap/supabase"
    echo "  npm:   npm install -g supabase"
    exit 1
fi

echo "✅ Supabase CLI found: $(supabase --version)"
echo ""

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY environment variable not set"
    echo ""
    read -p "Enter your Groq API key: " GROQ_KEY
    export GROQ_API_KEY=$GROQ_KEY
else
    echo "✅ GROQ_API_KEY found in environment"
fi

echo ""
echo "=========================================="
echo "Step 1: Initialize Supabase (if needed)"
echo "=========================================="
echo ""

if [ ! -f "supabase/config.toml" ]; then
    echo "Initializing Supabase project..."
    supabase init
    echo "✅ Supabase initialized"
else
    echo "✅ Supabase already initialized"
fi

echo ""
echo "=========================================="
echo "Step 2: Set Secrets"
echo "=========================================="
echo ""

echo "Setting GROQ_API_KEY secret..."
echo "$GROQ_API_KEY" | supabase secrets set GROQ_API_KEY --env-file /dev/stdin

echo "✅ Secrets configured"

echo ""
echo "=========================================="
echo "Step 3: Deploy Edge Function"
echo "=========================================="
echo ""

echo "Deploying evaluate-prescription function..."
supabase functions deploy evaluate-prescription

echo ""
echo "✅ Deployment complete!"

echo ""
echo "=========================================="
echo "Step 4: Get Function URL"
echo "=========================================="
echo ""

# Try to get project reference
PROJECT_REF=$(supabase projects list 2>/dev/null | grep -v "ID" | awk '{print $1}' | head -n 1)

if [ -n "$PROJECT_REF" ]; then
    FUNCTION_URL="https://${PROJECT_REF}.supabase.co/functions/v1/evaluate-prescription"
    echo "🎉 Your function is deployed at:"
    echo ""
    echo "   $FUNCTION_URL"
    echo ""
    echo "Update this URL in grpo_train_amr.py:"
    echo "   API_BASE_URL = \"$FUNCTION_URL\""
else
    echo "⚠️  Could not automatically detect project URL"
    echo ""
    echo "Get your function URL from:"
    echo "   https://app.supabase.com/project/_/functions"
    echo ""
    echo "Then update API_BASE_URL in grpo_train_amr.py"
fi

echo ""
echo "=========================================="
echo "Step 5: Test Function"
echo "=========================================="
echo ""

if [ -n "$FUNCTION_URL" ]; then
    echo "Testing function with sample request..."
    
    curl -X POST "$FUNCTION_URL" \
      -H "Content-Type: application/json" \
      -d '{
        "patient_case": {
          "patient_profile": {"age": 45, "sex": "M"},
          "diagnosis": "Test",
          "prescription": {"drug": "Test"}
        },
        "model_output": "Test output",
        "ground_truth": "Test reference",
        "metrics": ["clinical_accuracy"]
      }' 2>/dev/null | python -m json.tool || echo "Function deployed but test failed (this is normal if secrets aren't synced yet)"
    
    echo ""
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update API_BASE_URL in grpo_train_amr.py"
echo "2. Run: python test_pipeline.py"
echo "3. Run: python prepare_grpo_dataset.py"
echo "4. Run: python grpo_train_amr.py"
echo ""
echo "View function logs:"
echo "   supabase functions logs evaluate-prescription"
echo ""
echo "=========================================="

