Crypto Funding Rate Arbitrage Bot
A Python-based trading bot that capitalizes on cryptocurrency funding rate arbitrage opportunities by identifying and trading perpetual swap contracts with extreme funding rates on Phemex exchange.
🎯 Strategy Overview
This bot implements a funding rate mean reversion strategy by:

Monitoring Funding Rates: Continuously tracks funding rates across all available perpetual swap pairs on Phemex
Identifying Extremes: Selects the top 10 pairs with the highest positive and negative funding rates
Consistency Analysis: Measures how consistent these funding rates are over time to filter out noise and identify genuine arbitrage opportunities
Position Management: Opens positions based on available balance, automatically sizing trades appropriately
Risk Management: Implements automatic stop-loss (-0.1%) and take-profit (+0.1%) orders to protect capital

🔧 Key Features

Automated Trade Execution: Places limit orders at current market prices (bid/ask) for optimal fills
Dynamic Position Sizing: Calculates position sizes based on account balance and available margin
Leverage Management: Automatically sets leverage (default: 5x) for each trading pair
Risk Protection: Automatic SL/TP orders for all open positions
Error Handling: Robust exception handling to prevent crashes during execution
Real-time Monitoring: Fetches live order book data and market prices

📊 How It Works
1. Funding Rate Analysis
- Fetches funding rates for all perpetual pairs
- Ranks pairs by funding rate (positive and negative)
- Analyzes consistency using statistical methods
2. Trade Execution
- Selects top opportunities based on consistency score
- Calculates optimal position size per balance
- Places limit orders at best bid/ask prices
- Sets 5x leverage for capital efficiency
3. Position Protection
- Take Profit: +0.1% from entry price
- Stop Loss: -0.1% from entry price
- Both long and short positions supported
🚀 Getting Started
Prerequisites
bashpip install ccxt pandas numpy
Configuration
Set up your Phemex API credentials:
pythonphemex = ccxt.phemex({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True
})
Usage
python# Run the main strategy
python funding_arbitrage_bot.py
⚠️ Risk Disclaimer

High Risk: Cryptocurrency trading involves substantial risk of loss
Funding Rate Changes: Funding rates can change rapidly and unpredictably
Leverage Risk: Using leverage amplifies both gains and losses
Always test with small amounts first
Never invest more than you can afford to lose
