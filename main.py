from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import industries, stocks, portfolio_with_risk, portfolio_max_sharpe, \
                        portfolio_xtreme_variance, portfolio_risk_ranges, montecarlo

app = FastAPI(title="ProyectoGrad API")

app.include_router(industries.router, tags=["industries"])
app.include_router(stocks.router, tags=["stocks"])
app.include_router(portfolio_with_risk.router, tags=["portfolio"])
app.include_router(portfolio_max_sharpe.router, tags=["portfolio"])
app.include_router(portfolio_xtreme_variance.router, tags=["portfolio"])
app.include_router(portfolio_risk_ranges.router, tags=["portfolio"])

app.include_router(montecarlo.router, tags=["montecarlo"])

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # replace with specific origins in production, e.g. ["http://localhost:3000"]
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)