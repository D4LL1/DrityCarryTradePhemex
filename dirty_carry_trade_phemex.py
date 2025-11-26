import ccxt
import pandas as pd

api_keys = { "phemex": 
    { "apiKey": "179791ab-faf6-483d-9a21-eecbc50f8164",
      "secret": "YyWYEx9pRDLjJfzg4Rm4dKYYzobL03WhojlUmFXsLR1mNDZiNmY4OC1hMWYzLTQ0N2EtYTZhNy1hMzkzYzExNTg3M2M",
        "enableRateLimit": True,
          "options": {"defaultType": "swap"}
}}
phemex = ccxt.phemex(api_keys['phemex'])

def analyze_funding_rates_and_calculate_positions(phemex, api_keys):
    """
    Analyzes funding rates across perpetual swap markets and calculates position weights.
    
    Args:
        phemex: CCXT Phemex exchange instance
        api_keys: Dictionary containing API credentials
    
    Returns:
        residuals: DataFrame containing analysis results and position weights
    """
    
    # Initialize exchange and load markets
    
    markets = phemex.load_markets()
    funding = []
    
    # Fetch current funding rates for all swap contracts
    print("Fetching funding rates...")
    for symbol, market in phemex.markets.items():
        if market.get('swap') and market.get('contract'):
            market_id = market['id']
            try:
                rate = phemex.fetch_funding_rate(symbol)
                funding.append(rate)
            except Exception as e:
                print(f"Skipping {symbol} | ID: {market_id} | Error: {e}")
    
    # Create DataFrame and sort by funding rate
    df_funding = pd.DataFrame(funding)
    df_funding['fundingRate'] = pd.to_numeric(df_funding['fundingRate'], errors='coerce')
    df_funding = df_funding.sort_values(by='fundingRate')
    
    # Select top 10 negative (long candidates) and top 10 positive (short candidates)
    ticker_names = pd.concat([df_funding['symbol'].head(10), df_funding['symbol'].tail(10)])
    long_names = df_funding[['symbol']].head(10)
    short_names = df_funding[['symbol']].tail(10)
    
    # Fetch funding rate history for selected tickers
    print("Fetching funding rate history...")
    funding_list = []
    for ticker in ticker_names:
        try:
            funding_data = phemex.fetch_funding_rate_history(ticker)
            df_f = pd.DataFrame(funding_data)
            if 'timestamp' in df_f.columns:
                df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')
            df_f['symbol'] = ticker
            funding_list.append(df_f)
        except Exception as e:
            print(f"Skipping {ticker} funding history | Error: {e}")
    
    # Combine all funding history data
    df_funding_history = pd.concat(funding_list, ignore_index=True)
    
    # Convert numeric columns
    numeric_cols = ['fundingRate', 'funding', 'rate']
    for col in numeric_cols:
        if col in df_funding_history.columns:
            df_funding_history[col] = pd.to_numeric(df_funding_history[col], errors='coerce')
    
    # Calculate funding rate intervals for each symbol
    print("Calculating moving averages...")
    df_2 = df_funding_history.copy()
    interval_map = {}
    
    for symbol, group in df_2.groupby('symbol'):
        if len(group) > 1:
            td = group['timestamp'].iloc[1] - group['timestamp'].iloc[0]
            hours = td.total_seconds() / 3600
            interval_map[symbol] = hours
    
    df_2['interval'] = df_2['symbol'].map(interval_map)
    
    # Calculate 2-day exponential moving average
    df_2['2_days_moving_avr'] = (
        df_2.groupby('symbol')
            .apply(lambda g: g['fundingRate'].ewm(span=48 / g['interval'].iloc[0]).mean())
            .reset_index(level=0, drop=True)
    )
    
    # Calculate consistency metrics (R2, MSE, Accuracy)
    print("Calculating consistency metrics...")
    residuals = pd.DataFrame(index=["r2", "mse", "accuracy"], columns=ticker_names)
    
    for symbol in ticker_names:
        # Filter for this symbol
        df_sym = df_2[df_2["symbol"] == symbol]
        
        if len(df_sym) == 0:
            continue
        
        df_last = df_sym
        
        # R² calculation
        r2 = (
            ((df_last['2_days_moving_avr'].shift(1).dropna() - df_last['fundingRate'].mean()) ** 2).sum()
            /
            ((df_last['fundingRate'] - df_last['fundingRate'].mean()) ** 2).sum()
        )
        
        # MSE calculation
        mse = ((df_last['2_days_moving_avr'].shift(1) - df_last['fundingRate']) ** 2).mean()
        
        # Accuracy of sign prediction
        sign_ma = df_last['2_days_moving_avr'].shift(1).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        sign_funding = df_last['fundingRate'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        matches = (sign_ma == sign_funding).sum()
        accuracy = matches / len(df_last)
        
        # Save results
        residuals.loc["r2", symbol] = r2
        residuals.loc["mse", symbol] = mse
        residuals.loc["accuracy", symbol] = accuracy
    
    # Filter tickers with accuracy >= 80%
    residuals = residuals.loc[:, residuals.loc["accuracy"] >= 0.8]
    print(f"Filtered to {len(residuals.columns)} tickers with accuracy >= 80%")
    
    # Fetch account balance
    balance = phemex.fetch_balance()
    balance_for_shorts = balance['total']["USDT"] / 2
    balance_for_longs = balance['total']["USDT"] / 2
    
    # Assign BUY/SELL positions
    print("Assigning positions...")
    long_set = set(long_names['symbol'].dropna().values)
    common_tickers = [t for t in ticker_names if t in residuals.columns]
    
    positions = pd.Series(
        ["BUY" if t in long_set else "SELL" for t in common_tickers],
        index=common_tickers
    )
    
    residuals.loc["position", positions.index] = positions.values
    
    # Calculate position weights based on balance
    signals = residuals.loc["position"]
    is_buy = signals == "BUY"
    
    count_buy = is_buy.sum()
    count_sell = (~is_buy).sum()
    
    # Assign weights
    if count_buy > 0:
        residuals.loc["weight", is_buy] = balance_for_longs / count_buy - 1
    if count_sell > 0:
        residuals.loc["weight", ~is_buy] = balance_for_shorts / count_sell - 1
    
    print(f"\n✓ Analysis complete!")
    print(f"  - BUY signals: {count_buy}")
    print(f"  - SELL signals: {count_sell}")
    print(f"  - Balance per long: ${balance_for_longs / count_buy if count_buy > 0 else 0:.2f}")
    print(f"  - Balance per short: ${balance_for_shorts / count_sell if count_sell > 0 else 0:.2f}")
    
    return residuals

residuals = analyze_funding_rates_and_calculate_positions(phemex, api_keys)

# Usage example:
# residuals = analyze_funding_rates_and_calculate_positions(phemex, api_keys)
def place_residual_orders():
    """Place limit orders based on residuals DataFrame"""
    
    for ticker in residuals.columns:
        try:
            # Extract order details
            side = residuals[ticker][3]  # "BUY" or "SELL"
            amount = residuals[ticker][4]
            
            # Skip if no amount to trade
            if amount == 0:
                print(f"Skipping {ticker} - zero amount")
                continue
            
            # Set leverage
            phemex.set_leverage(symbol=ticker, leverage=5)
            
            # Fetch order book and determine price
            order_book = phemex.fetch_order_book(ticker)
            price = order_book["bids"][0][0] if side == "BUY" else order_book["asks"][0][0]
            
            # Place order using unified method
            order = phemex.create_order(
                symbol=ticker,
                type="limit",
                side=side.lower(),  # "buy" or "sell"
                amount=amount * 5 / price,  # Use actual amount instead of hardcoded 1
                price=price
            )
            
            print(f"Order placed → {ticker} | {side} | amount: {amount} | price: {price}")
            
        except Exception as e:
            print(f"Error placing order for {ticker}: {e}")
            continue

# Call the function
place_residual_orders()
def cancel_all_orders():
    for ticker in residuals.columns:
        open_orders = phemex.fetch_open_orders(symbol=ticker)

        for order in open_orders:
            try:
                phemex.cancel_order(order['id'], order['symbol'])
                print(f"Cancelled → {order['symbol']} | id: {order['id']}")
            except Exception as e:
                print(f"Error cancelling {order['id']}: {e}")


def set_stop_loss_take_profit_for_positions():
    """Set stop-loss (-0.1%) and take-profit (+0.1%) orders for all open positions"""
    positions = phemex.fetch_positions()

    for pos in positions:
        contracts = float(pos['contracts'])
        side = pos['side']  # 'long' or 'short'
        symbol = pos['symbol']

        if contracts == 0:
            continue  # empty position

        try:
            # Fetch current market price
            ticker = phemex.fetch_ticker(symbol)
            current_price = ticker['last']  # Current market price
            
            # Calculate stop-loss and take-profit prices based on position type
            if side == "long":
                # For long: TP above current, SL below current
                take_profit_price = current_price * 1.001  # +0.1%
                stop_loss_price = current_price * 0.999    # -0.1%
                close_side = "sell"
            else:  # short
                # For short: TP below current, SL above current
                take_profit_price = current_price * 0.999  # -0.1%
                stop_loss_price = current_price * 1.001    # +0.1%
                close_side = "buy"
            
            # Place Take Profit order (limit order to close position at profit)
            try:
                tp_order = phemex.create_order(
                    symbol=symbol,
                    type="limit",
                    side=close_side,
                    amount=contracts,
                    price=take_profit_price,
                    params={
                        "reduceOnly": True
                    }
                )
                print(f"✓ TP set → {symbol} | {side} | TP: {take_profit_price:.6f} (+0.1%)")
            except Exception as e:
                print(f"✗ Error setting TP for {symbol}: {e}")
            
            # Place Stop Loss order (conditional market order)
            try:
                sl_order = phemex.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=contracts,
                    params={
                        "trigger": stop_loss_price,  # Use trigger instead of stopLoss object
                        "reduceOnly": True
                    }
                )
                print(f"✓ SL set → {symbol} | {side} | SL: {stop_loss_price:.6f} (-0.1%)")
            except Exception as e:
                print(f"✗ Error setting SL for {symbol}: {e}")
                
        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")



    print("\nAll positions protected with SL/TP orders.")

def close_all_trades():
    print("Cancelling all open orders...")
    cancel_all_orders()

    print("\nSetting stop-loss and take-profit orders...")
    set_stop_loss_take_profit_for_positions()