#define MAGICMA  20150810

#define TRADE_TYPE_BUY 1
#define TRADE_TYPE_SELL 2
#define TRADE_TYPE_NOT_HAVE 3

#define MA_PERIOD 20

#define FALSE 0
#define TRUE 1

int touch_top(double ma_cur, double ma_prev1, double ma_prev2)
{
  if (Close[2] < ma_prev1 && Close[1] > ma_prev2 && Close[0] < ma_cur){
    return TRUE;
  } else {
    return FALSE;
  }
}

int touch_bottom(double ma_cur, double ma_prev1, double ma_prev2)
{
  if (Close[2] > ma_prev1 && Close[1] < ma_prev2 && Close[0] > ma_cur){
    return TRUE;
  } else {
    return FALSE;
  }
}

int cross_above(double ma_cur, double ma_prev1)
{
  if (Close[1] < ma_prev1 && Close[0] > ma_cur){
    return TRUE;
  } else {
    return FALSE;
  }  
}

int cross_below(double ma_cur, double ma_prev1)
{
  if (Close[1] > ma_prev1 && Close[0] < ma_cur){
    return TRUE;
  } else {
    return FALSE;
  }  
}

int check_active_order()
{
  for(int i = 0;i < OrdersTotal();i++) {
    if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES) == false){
      break;
    }
    if(OrderMagicNumber() != MAGICMA || OrderSymbol() != Symbol()){
      continue;
    }
    return TRUE;
  }
  return FALSE;
}

//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckForOpen()
{
   double ma_cur;
   double ma_prev1;
   double ma_prev2;
   
//---- go trading only for first tiks of new bar
   if(Volume[0] > 1){
     return;
   }
//---- get Moving Average 
   ma_cur = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 0);
   ma_prev1 = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 1);
   ma_prev2 = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 2);
   
//---- buy conditions
   if(touch_bottom(ma_cur, ma_prev1, ma_prev2)) {
     OrderSend(Symbol(), OP_SELL, NormalizeDouble(AccountFreeMargin()*0.1/1000.0,1), Bid, 3, 0, 0, "", MAGICMA, 0, Blue);
     return;
   }
//---- sell conditions
   if(touch_top(ma_cur, ma_prev1, ma_prev2)) {
     OrderSend(Symbol(), OP_BUY, NormalizeDouble(AccountFreeMargin()*0.1/1000.0,1), Ask, 3, 0, 0, "", MAGICMA, 0, Red); 
     return;
   }

//----
}
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
void CheckForClose()
{
   double ma_cur;
   double ma_prev1;
   double ma_prev2;
   
//---- go trading only for first tiks of new bar
   if(Volume[0] > 1) {
     return;
   }
//---- get Moving Average
   ma_cur = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 0);
   ma_prev1 = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 1);
   ma_prev2 = iMA(NULL, 0, MA_PERIOD, 0, MODE_SMA, PRICE_CLOSE, 2);   
//----
   for(int i = 0;i < OrdersTotal();i++)
     {
       if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES) == false){
	      break;
       }
       if(OrderMagicNumber() != MAGICMA || OrderSymbol() != Symbol()){
	 continue;
       }
       //---- check order type 
       if(OrderType() == OP_BUY)
	 {
	   if(cross_below(ma_cur, ma_prev1) || touch_bottom(ma_cur, ma_prev1, ma_prev2) || OrderProfit() > AccountFreeMargin()*(-0.07)){
	     OrderClose(OrderTicket(), OrderLots(), Bid, 3, White);
	   }
	   break;
        }
      if(OrderType() == OP_SELL)
        {
	  if(cross_above(ma_cur, ma_prev1) || touch_top(ma_cur, ma_prev1, ma_prev2)|| OrderProfit() > AccountFreeMargin()*(-0.07)){
	    OrderClose(OrderTicket(), OrderLots(), Ask, 3, White);
	  }  
	  break;
        }
     }
//----
}
//+------------------------------------------------------------------+
//| Start function                                                   |
//+------------------------------------------------------------------+
void start()
{
//---- check for history and trading
    if (Bars < 100 || IsTradeAllowed() == false){
      return;
    }
//---- calculate open orders by current symbol

    if (check_active_order() == FALSE){
      CheckForOpen();
    } else {
      CheckForClose();
    }
//----
}
//+------------------------------------------------------------------+
