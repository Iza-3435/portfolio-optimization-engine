from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
import time
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure CORS more explicitly
CORS(app, 
     origins=["*"],  # Allow all origins for development
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Cache for stock data to avoid excessive requests
stock_cache = {}
CACHE_DURATION = timedelta(minutes=15)

print(f"🚀 Starting Portfolio Intelligence Backend")
print(f"💰 Using Yahoo Finance (yfinance) - FREE unlimited data!")
print(f"🌐 CORS enabled for all origins")

class YahooFinanceClient:
    def __init__(self):
        print("💰 Yahoo Finance client initialized - no API key needed!")
        
    def get_quote(self, symbol):
        """Get current quote for a symbol"""
        cache_key = f"quote_{symbol}"
        
        # Check cache first
        if cache_key in stock_cache:
            cache_time, cache_data = stock_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=5):
                print(f"📋 Using cached quote for {symbol}")
                return cache_data
        
        try:
            # Get stock info from Yahoo Finance
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2d")  # Get last 2 days for change calculation
            
            if len(hist) == 0:
                raise ValueError(f"No data found for symbol {symbol}")
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate change
            if len(hist) >= 2:
                prev_price = float(hist['Close'].iloc[-2])
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100
            else:
                change = 0
                change_percent = 0
            
            result = {
                'symbol': symbol.upper(),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': f"{change_percent:.2f}",  # Fixed formatting
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
            }
            
            # Cache the result
            stock_cache[cache_key] = (datetime.now(), result)
            print(f"💰 Got quote for {symbol}: ${result['price']:.2f} ({result['change_percent']}%)")
            return result
            
        except Exception as e:
            print(f"❌ Error fetching quote for {symbol}: {e}")
            raise ValueError(f"Could not fetch quote for {symbol}")
    
    def get_daily_data(self, symbol, period="1y"):
        """Get daily historical data for a symbol"""
        cache_key = f"daily_{symbol}_{period}"
        
        # Check cache first
        if cache_key in stock_cache:
            cache_time, cache_data = stock_cache[cache_key]
            if datetime.now() - cache_time < CACHE_DURATION:
                print(f"📋 Using cached data for {symbol}")
                return cache_data
        
        try:
            print(f"🔄 Downloading data for {symbol} from Yahoo Finance...")
            
            # Get historical data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, auto_adjust=True, prepost=False)
            
            if len(df) == 0:
                raise ValueError(f"No historical data found for symbol {symbol}")
            
            # Ensure we have the expected columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing {col} data for {symbol}")
            
            # Clean the data
            df = df.dropna()
            
            if len(df) < 30:
                raise ValueError(f"Insufficient data for {symbol} (only {len(df)} days)")
            
            # IMPORTANT: Convert timezone-aware datetime to timezone-naive
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Cache the result
            stock_cache[cache_key] = (datetime.now(), df)
            print(f"💾 Cached data for {symbol} ({len(df)} days)")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            raise
    
    def search_symbol(self, keywords):
        """Search for symbols"""
        # Common stocks mapping for better search
        common_stocks = {
            'apple': {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            'microsoft': {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            'google': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            'alphabet': {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            'amazon': {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            'tesla': {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            'nvidia': {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            'meta': {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            'facebook': {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            'netflix': {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
            'disney': {'symbol': 'DIS', 'name': 'Walt Disney Company'},
            'coca': {'symbol': 'KO', 'name': 'Coca-Cola Company'},
            'pepsi': {'symbol': 'PEP', 'name': 'PepsiCo Inc.'},
            'johnson': {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
            'jpmorgan': {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            'visa': {'symbol': 'V', 'name': 'Visa Inc.'},
            'walmart': {'symbol': 'WMT', 'name': 'Walmart Inc.'},
            'home': {'symbol': 'HD', 'name': 'Home Depot Inc.'},
            'boeing': {'symbol': 'BA', 'name': 'Boeing Company'},
            'intel': {'symbol': 'INTC', 'name': 'Intel Corporation'},
            'amd': {'symbol': 'AMD', 'name': 'Advanced Micro Devices'},
        }
        
        keywords_lower = keywords.lower()
        matches = []
        
        # Search by name
        for key, stock_info in common_stocks.items():
            if keywords_lower in key or key in keywords_lower:
                matches.append({
                    'symbol': stock_info['symbol'],
                    'name': stock_info['name'],
                    'type': 'Equity',
                    'region': 'United States',
                    'currency': 'USD'
                })
        
        # Also try direct symbol match
        keywords_upper = keywords.upper()
        if len(keywords_upper) <= 5:  # Likely a stock symbol
            try:
                # Test if the symbol exists by trying to get basic info
                test_stock = yf.Ticker(keywords_upper)
                test_hist = test_stock.history(period="1d")
                if len(test_hist) > 0:
                    # Try to get company name
                    try:
                        info = test_stock.info
                        name = info.get('longName', info.get('shortName', f'{keywords_upper} Corporation'))
                    except:
                        name = f'{keywords_upper} Corporation'
                    
                    # Add to beginning of results if it's a valid symbol
                    matches.insert(0, {
                        'symbol': keywords_upper,
                        'name': name,
                        'type': 'Equity',
                        'region': 'United States',
                        'currency': 'USD'
                    })
            except Exception as e:
                print(f"⚠️  Symbol test failed for {keywords_upper}: {e}")
        
        # Remove duplicates
        seen_symbols = set()
        unique_matches = []
        for match in matches:
            if match['symbol'] not in seen_symbols:
                seen_symbols.add(match['symbol'])
                unique_matches.append(match)
        
        print(f"🔍 Found {len(unique_matches)} matches for '{keywords}'")
        return unique_matches[:5]  # Return top 5

# Initialize Yahoo Finance client
yahoo_client = YahooFinanceClient()

class PortfolioOptimizer:
    def __init__(self, stocks, start_date=None, end_date=None):
        self.stocks = stocks
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=365))
        self.data = None
        self.returns = None
        
    def fetch_data(self):
        """Fetch historical price data for all stocks using Yahoo Finance"""
        print(f"📈 Fetching data for {len(self.stocks)} stocks: {', '.join(self.stocks)}")
        prices = pd.DataFrame()
        failed_stocks = []
        
        for i, symbol in enumerate(self.stocks):
            try:
                print(f"📊 Fetching data for {symbol} ({i+1}/{len(self.stocks)})...")
                
                # Get daily data from Yahoo Finance
                df = yahoo_client.get_daily_data(symbol, period="1y")
                
                if df is None or len(df) == 0:
                    print(f"❌ No data returned for {symbol}")
                    failed_stocks.append(symbol)
                    continue
                
                # Handle timezone-aware datetime comparison safely
                try:
                    # Make start_date timezone-naive to match df.index
                    start_date_to_use = self.start_date
                    if hasattr(start_date_to_use, 'tzinfo') and start_date_to_use.tzinfo is not None:
                        start_date_to_use = start_date_to_use.replace(tzinfo=None)
                    
                    if start_date_to_use:
                        df_filtered = df[df.index >= start_date_to_use]
                    else:
                        df_filtered = df
                except Exception as date_error:
                    print(f"⚠️  Date filtering failed for {symbol}: {date_error}")
                    df_filtered = df
                
                if len(df_filtered) > 30:  # Need at least 30 days
                    prices[symbol] = df_filtered['Close']
                    print(f"✅ Got {len(df_filtered)} data points for {symbol}")
                else:
                    print(f"⚠️  Insufficient filtered data for {symbol} ({len(df_filtered)} days)")
                    # Use full dataset if filtering resulted in too little data
                    if len(df) > 30:
                        prices[symbol] = df['Close']
                        print(f"✅ Using full dataset for {symbol} ({len(df)} days)")
                    else:
                        failed_stocks.append(symbol)
                        
            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")
                failed_stocks.append(symbol)
                continue
        
        if failed_stocks:
            print(f"⚠️  Failed to fetch data for: {', '.join(failed_stocks)}")
        
        if prices.empty:
            raise ValueError("No data could be fetched for any symbols")
        
        if len(prices.columns) < 2:
            raise ValueError(f"Need at least 2 stocks with valid data. Got: {', '.join(prices.columns)}")
            
        # Handle missing values
        prices_clean = prices.fillna(method='ffill').fillna(method='bfill').dropna()
        
        if len(prices_clean) < 30:
            raise ValueError(f"Insufficient data points ({len(prices_clean)} days)")
        
        print(f"🎯 Successfully processed data for {len(prices_clean.columns)} stocks with {len(prices_clean)} data points each")
        
        self.data = prices_clean
        self.returns = prices_clean.pct_change().dropna()
        
        if self.returns.empty or self.returns.isnull().any().any():
            self.returns = self.returns.dropna()
        
        print(f"📈 Returns data: {len(self.returns)} periods for {len(self.returns.columns)} stocks")
        
    def calculate_metrics(self, weights):
        """Calculate portfolio metrics given weights"""
        weights = np.array(weights)
        
        # Annual returns and volatility
        portfolio_returns = self.returns @ weights
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        
        # Beta calculation (using SPY as market proxy)
        try:
            spy_data = yahoo_client.get_daily_data('SPY', period="1y")
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            if len(common_dates) > 0:
                portfolio_aligned = portfolio_returns[common_dates]
                spy_aligned = spy_returns[common_dates]
                covariance = np.cov(portfolio_aligned, spy_aligned)[0, 1]
                spy_variance = np.var(spy_aligned)
                beta = covariance / spy_variance if spy_variance > 0 else 1.0
            else:
                beta = 1.0
        except:
            beta = 1.0
            
        return {
            'expected_return': annual_return * 100,
            'volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'var_95': var_95 * 100,
            'beta': beta
        }
    
    def optimize_portfolio(self, method='max_sharpe', max_weight=0.4, min_weight=0.05):
        """Optimize portfolio weights with constraints"""
        n = len(self.stocks)
        print(f"🎯 Optimizing portfolio using method: {method}")
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds with min/max weight constraints for diversification
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        
        # Initial guess: equal weights
        x0 = np.array([1/n] * n)
        
        if method == 'max_sharpe':
            def neg_sharpe_with_diversification(weights):
                try:
                    portfolio_returns = self.returns @ weights
                    annual_return = portfolio_returns.mean() * 252
                    annual_volatility = portfolio_returns.std() * np.sqrt(252)
                    risk_free_rate = 0.02
                    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                    
                    # Add diversification penalty
                    hhi = np.sum(weights ** 2)
                    diversification_penalty = hhi * 2
                    
                    return -(sharpe_ratio - diversification_penalty)
                except:
                    return 1000
            
            result = minimize(neg_sharpe_with_diversification, x0, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            
        elif method == 'min_volatility':
            def portfolio_volatility(weights):
                try:
                    portfolio_returns = self.returns @ weights
                    return portfolio_returns.std() * np.sqrt(252)
                except:
                    return 1000
            
            result = minimize(portfolio_volatility, x0, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
        else:  # equal_weight
            result = type('obj', (object,), {'x': x0, 'success': True})()
            
        if result.success:
            final_weights = result.x.copy()
            final_weights[final_weights < 0.01] = 0
            final_weights = final_weights / np.sum(final_weights)
            return final_weights
        else:
            return np.array([1/n] * n)
    
    def get_technical_signals(self):
        """Calculate simple technical indicators for each stock"""
        signals = []
        
        for symbol in self.stocks:
            if symbol not in self.data.columns:
                continue
                
            prices = self.data[symbol]
            
            # Calculate RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Calculate SMA trend
            if len(prices) >= 50:
                sma_20 = prices.rolling(window=20).mean()
                sma_50 = prices.rolling(window=50).mean()
                sma_trend = 'UP' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'DOWN'
            else:
                sma_trend = 'UNKNOWN'
            
            # Generate signal
            if current_rsi < 30:
                signal = 'BUY'
            elif current_rsi > 70:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'rsi': round(float(current_rsi), 1),
                'sma_trend': sma_trend
            })
            
        return signals
    
    def monte_carlo_simulation(self, weights, num_simulations=1000, days=252):
        """Run Monte Carlo simulation for portfolio"""
        weights = np.array(weights)
        
        # Calculate portfolio statistics
        portfolio_returns = self.returns @ weights
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Initial portfolio value
        initial_value = 10000
        
        # Run simulations
        simulations = np.zeros((num_simulations, days))
        
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, days)
            
            # Calculate cumulative value
            cumulative_returns = np.cumprod(1 + random_returns)
            simulations[i] = initial_value * cumulative_returns
            
        # Calculate percentiles
        percentiles = {
            'p5': np.percentile(simulations, 5, axis=0).tolist(),
            'p50': np.percentile(simulations, 50, axis=0).tolist(),
            'p95': np.percentile(simulations, 95, axis=0).tolist()
        }
        
        return percentiles

    def get_correlation_matrix(self):
        """Calculate actual correlation matrix from returns data"""
        try:
            if self.returns is None or self.returns.empty:
                return None
            
            if len(self.returns.columns) < 2:
                return None
            
            correlation_matrix = self.returns.corr()
            
            # Check for NaN values in correlation matrix
            if correlation_matrix.isnull().any().any():
                correlation_matrix = correlation_matrix.fillna(0)
            
            # Convert to the format expected by frontend
            correlations = []
            stocks = list(correlation_matrix.columns)
            
            for i, stock1 in enumerate(stocks):
                for j, stock2 in enumerate(stocks):
                    corr_value = correlation_matrix.loc[stock1, stock2]
                    if pd.isna(corr_value):
                        corr_value = 0.0
                    
                    correlations.append({
                        'x': i,
                        'y': j,
                        'v': float(corr_value),
                        'stock1': stock1,
                        'stock2': stock2
                    })
            
            return {
                'correlations': correlations,
                'stocks': stocks,
                'matrix': correlation_matrix.round(3).to_dict()
            }
            
        except Exception as e:
            print(f"❌ Error calculating correlation matrix: {e}")
            return None

# Add OPTIONS handling for CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    result = {
        'status': 'healthy',
        'data_source': 'Yahoo Finance (yfinance)',
        'api_key_required': False,
        'unlimited_requests': True,
        'real_time_delay': '15-20 minutes',
        'timestamp': datetime.now().isoformat(),
        'server': 'Flask Portfolio Intelligence'
    }
    print(f"🏥 Health check requested - Yahoo Finance ready")
    return jsonify(result)

@app.route('/api/search/<keywords>')
def search_stocks(keywords):
    """Search for stock symbols"""
    print(f"🔍 Search requested for: {keywords}")
    try:
        results = yahoo_client.search_symbol(keywords)
        return jsonify({'results': results})
    except Exception as e:
        print(f"❌ Search error for '{keywords}': {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-stock/<symbol>')
def validate_stock(symbol):
    """Validate if a stock symbol exists and get current price"""
    print(f"✅ Validation requested for: {symbol}")
    try:
        quote = yahoo_client.get_quote(symbol)
        result = {
            'valid': True,
            'symbol': quote['symbol'],
            'price': quote['price'],
            'change': quote['change'],
            'change_percent': quote['change_percent'],
            'volume': quote['volume']
        }
        print(f"✅ Validation successful for {symbol}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Validation failed for {symbol}: {e}")
        return jsonify({
            'valid': False, 
            'symbol': symbol,
            'error': str(e)
        }), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    """Main analysis endpoint"""
    print(f"📊 Portfolio analysis requested")
    try:
        data = request.json
        stocks = data.get('stocks', [])
        optimization_method = data.get('optimization_method', 'max_sharpe')
        
        print(f"📋 Analysis request: {len(stocks)} stocks, method: {optimization_method}")
        print(f"📋 Stocks: {', '.join(stocks)}")
        
        if len(stocks) < 2:
            return jsonify({'error': 'At least 2 stocks required'}), 400
            
        # Initialize optimizer
        optimizer = PortfolioOptimizer(stocks)
        
        # Fetch data
        try:
            optimizer.fetch_data()
        except Exception as e:
            print(f"❌ Data fetching failed: {e}")
            return jsonify({'error': f'Failed to fetch stock data: {str(e)}'}), 500
        
        if optimizer.returns.empty:
            return jsonify({'error': 'No valid data found for the selected stocks'}), 500
            
        # Optimize portfolio
        optimal_weights = optimizer.optimize_portfolio(optimization_method)
        
        # Calculate metrics
        metrics = optimizer.calculate_metrics(optimal_weights)
        
        # Get allocation
        allocation = [
            {'symbol': symbol, 'weight': round(weight * 100, 2)}
            for symbol, weight in zip(stocks, optimal_weights)
        ]
        
        # Get performance data
        portfolio_returns = optimizer.returns @ optimal_weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = 10000 * cumulative_returns
        
        performance = {
            'dates': portfolio_values.index.strftime('%Y-%m-%d').tolist(),
            'values': portfolio_values.round(2).tolist()
        }
        
        # Get technical signals
        signals = optimizer.get_technical_signals()
        
        # Run Monte Carlo simulation
        monte_carlo = {
            'percentiles': optimizer.monte_carlo_simulation(optimal_weights)
        }
        
        # Get correlation matrix
        correlation_data = optimizer.get_correlation_matrix()
        
        result = {
            'metrics': metrics,
            'allocation': allocation,
            'performance': performance,
            'signals': signals,
            'monte_carlo': monte_carlo,
            'correlation_data': correlation_data,
            'data_points': len(optimizer.data)
        }
        
        print(f"✅ Analysis completed successfully with {len(optimizer.data)} data points")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    print("🧪 Test endpoint called")
    return jsonify({
        'message': 'Backend is working!',
        'timestamp': datetime.now().isoformat(),
        'cors_enabled': True,
        'server': 'Flask Portfolio Intelligence',
        'data_source': 'Yahoo Finance (yfinance)',
        'port': 5000
    })

@app.route('/')
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Portfolio Intelligence API',
        'status': 'running',
        'data_source': 'Yahoo Finance (yfinance)',
        'endpoints': [
            '/api/health',
            '/api/test',
            '/api/search/<keywords>',
            '/api/validate-stock/<symbol>',
            '/api/analyze'
        ]
    })

if __name__ == '__main__':
    print(f"🚀 Starting Flask server...")
    print(f"💰 Using Yahoo Finance - FREE unlimited stock data!")
    print(f"⏱️  Real-time data with 15-20 minute delay")
    print(f"🌐 Server will be available at: http://127.0.0.1:5000")
    print(f"🧪 Test endpoint: http://127.0.0.1:5000/api/test")
    print(f"🏥 Health check: http://127.0.0.1:5000/api/health")
    
    app.run(debug=True, port=5000, host='127.0.0.1', threaded=True)