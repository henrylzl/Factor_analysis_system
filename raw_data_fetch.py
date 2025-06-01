import os
import numpy as np
import pandas as pd
import tushare as ts
from retrying import retry
from functools import wraps
from factor_generate import FactorGenerater
from dotenv import load_dotenv

#打印能完整显示
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 50000)
pd.set_option('max_colwidth', 1000)

class RawDataFetcher(FactorGenerater):
    # 计算给定日期的当月最后一天
    def _get_month_end(self, date):
        """计算给定日期的当月最后一天
        Args:
            date (datetime): 需要计算的日期对象
        Returns:
            datetime: 当月最后一天的日期对象

        逻辑说明：
        1. 如果当天就是月末，返回原日期
        2. 如果不是，通过MonthEnd偏移量计算月末日期
        """
        import calendar
        import pandas.tseries.offsets as toffsets
        _, days = calendar.monthrange(date.year, date.month)
        if date.day == days:
            return date
        else:
            return date + toffsets.MonthEnd(n=1)

    # 确保按交易日获取数据
    @retry(stop_max_attempt_number=500, wait_random_min=1000, wait_random_max=2000)
    def ensure_data(self, func, save_dir, start_dt='20140101', end_dt='20250430'):
        """确保按交易日获取数据
        Args:
            func: 数据获取函数，接受日期字符串作为参数（格式：YYYYMMDD）
            save_dir: 数据保存目录名称
            start_dt: 起始日期（默认2014-01-01）
            end_dt: 结束日期（默认2025-04-30）

        功能：
        1. 自动创建存储目录
        2. 检查本地已有数据文件
        3. 遍历交易日下载缺失数据
        4. 自动重试失败请求（通过retry装饰器）
        """
        tmp_dir = os.path.join(self.root, save_dir)
        os.makedirs(tmp_dir, exist_ok=True) # 创建目标目录（如果不存在）
        dl = [pd.to_datetime(name.split(".")[0]) for name in os.listdir(tmp_dir)]
        dl = sorted(dl)
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        tdays = pd.Series(self.tradedays, index=self.tradedays)
        tdays = tdays[(tdays>=s)&(tdays<=e)]
        tdays = tdays.index.tolist()
        for tday in tdays:
            if tday in dl: continue
            t = tday.strftime("%Y%m%d")
            try:
                datdf = func(t)
                path = os.path.join(tmp_dir, t + ".csv")
                datdf.to_csv(path, encoding='gbk')
                print(f"{t}.csv 写入成功！")
            except Exception as e:
                print(f"获取 {t} 数据失败: {str(e)}")
                # 触发retry机制（最多重试500次，间隔1-2秒）
                raise
        if save_dir == '__hs300_wt__' or '__zz500_wt__':
            self.clean_empty_files(save_dir)
            print(f"{save_dir}空文件清理完成")
            column_mapping = {'con_code': 'ts_code'}
            result = self.rename_columns_in_folder(save_dir, column_mapping)
            print("\n处理结果:")
            print(f"成功处理 {result['成功']} 个文件")
            if result['失败']:
                print(f"处理失败的文件: {result['失败']}")

    # 确保按月频（上月末及当月初）获取数据
    @retry(stop_max_attempt_number=500, wait_random_min=1000, wait_random_max=2000)
    def ensure_data_by_m(self, func, save_dir, start_dt='20140101', end_dt='20250430'):
        """确保按月获取数据
        Args:
            func: 数据获取函数，接受日期字符串作为参数（格式：YYYYMMDD）
            save_dir: 数据保存目录名称
            start_dt: 起始日期（默认2014-01-01）
            end_dt: 结束日期（默认2025-04-30）

        功能：
        1. 自动创建存储目录
        2. 检查本地已有数据文件
        3. 遍历交易日下载缺失数据
        4. 自动重试失败请求（通过retry装饰器）
        """
        tmp_dir = os.path.join(self.root, save_dir)
        os.makedirs(tmp_dir, exist_ok=True) # 创建目标目录（如果不存在）
        dl = [pd.to_datetime(name.split(".")[0]) for name in os.listdir(tmp_dir)]
        dl = sorted(dl)
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        # 将self.trade_days_begin_end_of_month的两个列合并成一列
        mdayb = pd.Series(self.trade_days_begin_end_of_month['month_start'])
        mdaye = pd.Series(self.trade_days_begin_end_of_month.index)
        mdays = pd.concat([mdayb, mdaye], ignore_index=True)
        mdays =  mdays.sort_values()
        mdays = pd.Series(mdays.values, index=mdays.values)
        mdays = mdays[(mdays>=s)&(mdays<=e)]
        mdays = mdays.index.tolist()
        for tday in mdays:
            if tday in dl: continue
            t = tday.strftime("%Y%m%d")
            try:
                datdf = func(t)
                path = os.path.join(tmp_dir, t + ".csv")
                datdf.to_csv(path, encoding='gbk')
                print(f"{t}.csv 写入成功！")
            except Exception as e:
                print(f"获取 {t} 数据失败: {str(e)}")
                # 触发retry机制（最多重试500次，间隔1-2秒）
                raise

    # 确保按季度获取数据
    @retry(stop_max_attempt_number=500, wait_random_min=1000, wait_random_max=2000)
    def ensure_data_by_q(self, func, save_dir, start_dt='20140101', end_dt='20250430'):
        """确保按季度获取并更新数据
        Args:
            func: 数据获取函数，接受period参数（格式：YYYYMMDD）
            save_dir: 数据保存目录名称
            start_dt: 起始日期（默认2014-01-01）
            end_dt: 结束日期（默认2025-04-30）

        功能特点：
        1. 保留最近3个季度的更新能力（强制重新下载最后三个季度数据）
        2. 自动创建存储目录
        3. 按季度频率遍历下载数据
        4. 自动重试失败请求（通过retry装饰器）
        """
        tmp_dir = os.path.join(self.root, save_dir)
        os.makedirs(tmp_dir, exist_ok=True) # 创建目标目录（如果不存在）
        dl = [pd.to_datetime(name.split(".")[0]) for name in os.listdir(tmp_dir)]
        dl = sorted(dl)
        if len(dl) > 3:
            dl = dl[0:len(dl)-3] #已经存在的最后三个季度数据重新下载
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        qdates = pd.date_range(start=s, end=e, freq='Q')
        qdates = qdates.tolist()
        for tday in qdates:
            if tday in dl: continue
            t = tday.strftime("%Y%m%d")
            try:
                datdf = func(period=t)
                path = os.path.join(tmp_dir, t + ".csv")
                datdf.to_csv(path, encoding='gbk')
                print(f"{t}.csv 写入成功！")
            except Exception as e:
                print(f"获取 {t} 季度数据失败: {str(e)}")
                # 触发retry机制（最多重试500次，间隔1-2秒）
                raise

    # 通过日频数据创建日频指标
    def create_indicator(self, raw_data_dir, raw_data_field, indicator_name):
        ''' 主要用于通过日频数据创建日频指标
        Args:
            raw_data_dir: 原始数据存储目录名称
            raw_data_field: 需要提取的原始数据字段
            indicator_name: 生成的指标名称（用于保存）

        功能特点：
        1. 遍历指定目录下的所有CSV文件
        2. 提取每个文件中的指定字段
        3. 构建新的DataFrame，以股票代码为索引，日期为列
        4. 保存生成的指标数据
        '''
        tmp_dir = os.path.join(self.root, raw_data_dir)
        os.makedirs(tmp_dir, exist_ok=True) # 创建目标目录（如果不存在）
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays)
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col=['ts_code'], engine='python', encoding='gbk')
            df[tday] = dat[raw_data_field]
            print(tday)
        df = df.dropna(how='all') #删掉全为空的一行
        diff = df.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
        df = df.drop(labels=diff)
        self.close_file(df, indicator_name)

    # 通过日频数据创建月频指标
    def create_indicator_m_by_d(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20140101', end_dt='20250430'):
        ''' 通过日频数据创建月频指标
        Args:
            raw_data_dir: 原始数据存储目录名称
            raw_data_field: 需要提取的数据字段
            indicator_name: 生成的指标名称
            start_dt: 起始日期（默认2014-01-01）
            end_dt: 结束日期（默认2025-04-30）

        功能特点：
        1. 遍历指定目录下的所有CSV文件
        2. 提取每个文件中的指定字段
        3. 构建新的DataFrame，以股票代码为索引，日期为列
        4. 保存生成的指标数据
        '''
        tmp_dir = os.path.join(self.root, raw_data_dir)
        os.makedirs(tmp_dir, exist_ok=True) # 创建目标目录（如果不存在）
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat[raw_data_field]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    # 通过日频数据创建月频指标（扩展版本）
    def create_indicator_m_by_d_ex(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20140101', end_dt='20250430'):
        ''' 通过日频数据创建月频指标（扩展版本）
        Args:
            raw_data_dir: 原始数据存储目录
            raw_data_field: 目标数据字段名
            indicator_name: 生成指标名称
            start_dt: 起始日期（默认2014-01-01）
            end_dt: 结束日期（默认2025-04-30）
        '''
        self.create_indicator(raw_data_dir, raw_data_field, indicator_name)
        datdf = getattr(self, indicator_name, None)
        datdf = self.preprocess(datdf)
        self.close_file(datdf, indicator_name)
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            caldate = self.month_map[tday]
            df[caldate] = datdf[tday]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name+'_m')

    # 通过季频数据创建月频指标
    def create_indicator_m_by_q(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20140101', end_dt='20250430'):
        ''' 通过季频数据创建月频指标,主要用于财报数据处理
        Args:
            raw_data_dir: 财报数据存储目录
            raw_data_field: 目标财务字段名
            indicator_name: 生成指标名称
            start_dt: 统计起始日期（默认2014-01-01）
            end_dt: 统计结束日期（默认2025-04-30）
        '''
        s = pd.to_datetime(start_dt) #统计周期开始
        e = pd.to_datetime(end_dt) #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天（财报发布日期）
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天（指标日期）
        all_stocks_info = self.meta
        tmp_dir = os.path.join(self.root, raw_data_dir) #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk', parse_dates=['ann_date','end_date']) # 解析公告日期和财报截止日期
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0'] # 删除冗余列
            panel[d] = dat
            print(d)
        # 转换数据格式为三维面板（季度×股票×字段）
        datpanel = pd.concat(panel, axis=0)
        datpanel = datpanel.stack().unstack(level=(1, -1))
        #开始计算结果指标(月频),在每个时间截面逐个处理每只股票
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            for stock in df.index: #每只股票
                try:
                    datdf = datpanel[stock]
                    datdf = datdf.loc[datdf['ann_date']<d] #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
                    df.at[stock, d] = datdf.iloc[-1].at[raw_data_field] #取已经发布最近一期财报数据指定字段进行赋值
                    #print(stock)
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    #通过季频数据创建月频指标（扩展版本）
    def create_indicator_m_by_q_ex(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20140101', end_dt='20250430'):
        ''' 通过季频数据创建月频指标（扩展版本），主要用于财报数据处理
        Args:
            raw_data_dir: 财报数据存储目录
            raw_data_field: 目标财务字段名
            indicator_name: 生成指标名称
            start_dt: 统计起始日期（默认2014-01-01）
            end_dt: 统计结束日期（默认2025-04-30）
        '''
        s = pd.to_datetime(start_dt) #统计周期开始
        e = pd.to_datetime(end_dt) #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天
        all_stocks_info = self.meta
        tmp_dir = os.path.join(self.root, raw_data_dir) #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        # 使用concat将字典转换为多层索引的DataFrame
        datpanel = pd.concat(panel, axis=0)
        # 重新排列索引层次,使股票代码成为第一层索引
        datpanel = datpanel.stack().unstack(level=(1, -1))
        #开始计算结果指标(月频),在每个时间截面逐个处理每只股票
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            for stock in df.index: #每只股票
                try:
                    datdf = datpanel.loc[stock]
                    datdf = datdf.loc[datdf['ann_date']<d] #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
                    df.at[stock, d] = datdf.iloc[-1].at[raw_data_field] #取已经发布最近一期财报数据指定字段进行赋值
                    #print(stock)
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    # 对齐股票和时间
    def _align_element(self, df1, df2):
        ''' 对齐股票和时间
        Args:
            df1 (pd.DataFrame): 第一个数据集，索引为股票代码，列为日期
            df2 (pd.DataFrame): 第二个数据集，索引为股票代码，列为日期
        Returns:
            tuple: (对齐后的df1子集, 对齐后的df2子集)

        实现步骤:
            1. 获取两个数据集股票代码的交集，并按字母顺序排序
            2. 获取两个数据集日期列的交集，并按时间顺序排序
            3. 使用双重索引对齐方式截取公共数据集
        '''
        row_index = sorted(df1.index.intersection(df2.index))
        col_index = sorted(df1.columns.intersection(df2.columns))
        return df1.loc[row_index, col_index], df2.loc[row_index, col_index]

    # 生成每日行情指标
    def create_daily_quote_indicators(self):
        ''' 生成每日行情指标
        Args:
            无显式参数，通过类属性获取数据
        Returns:
            无直接返回值，通过close_file方法存储生成指标

        实现功能：
        1. 创建基础行情指标（复权因子/成交量/收盘价等）
        2. 补充三大指数数据
        3. 生成不同周期的收益率指标
        实现步骤：
        - 第一部分：创建基础指标并进行数据预处理
        - 第二部分：整合指数数据到股票数据集
        - 第三部分：计算月频收益率指标（1/3/6/12个月）
        '''
        #-------------------------------------------------------------
        #创建一些行情指标
        # 生成复权因子指标（从原始日频数据）
        self.create_indicator("__temp_adj_factor__", "adj_factor", "adjfactor")
        adjfactor = self.preprocess(self.adjfactor)
        self.close_file(adjfactor, 'adjfactor')

        # 生成成交量指标（单位转换：千元->万元）
        self.create_indicator("__temp_daily__", "amount", "amt")
        amt = self.amt / 10 #默认每单位千元,转换为每单位万元
        amt = self.preprocess(amt, suspend_days_process=True, val=0) # 停牌日成交量设为0
        self.close_file(amt, 'amt')

        # 生成收盘价指标（从原始日频数据）
        self.create_indicator("__temp_daily__", "close", "close")
        close = self.preprocess(self.close) # 预处理原始收盘价
        self.close_file(close, 'close')

        # 计算后复权收盘价（基于对齐后的收盘价和复权因子）
        close, adjfactor = self._align_element(self.close, self.adjfactor)
        hfq_close = close * adjfactor
        self.close_file(hfq_close, 'hfq_close') #后复权收盘价

        # 生成涨跌幅指标（百分比转换为小数）
        self.create_indicator("__temp_daily__", "pct_chg", "pct_chg")
        pct_chg = self.preprocess(self.pct_chg, suspend_days_process=True, val=0) # 停牌日涨跌幅设为0
        self.close_file(pct_chg, 'pct_chg')
        #-------------------------------------------------------------
        #将三大指数的数据给补上
        pct_chg = self.pct_chg # 获取预处理后的涨跌幅数据
        close = self.close # 获取预处理后的收盘价数据
        hfq_close = self.hfq_close # 获取后复权价格数据
        benchmarks = ['000001.SH', '000300.SH', '000905.SH'] #上证综指,沪深300,中证500
        tmp_dir = os.path.join(self.root, "__temp_index_daily__")
        # 遍历三大指数
        for name in benchmarks:
            # 读取指数日线数据文件
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[2], engine='python', encoding='gbk', parse_dates=['trade_date'])
            pct_chg.loc[name] = dat['pct_chg'][pct_chg.columns] # 将指数涨跌幅数据添加到pct_chg 
            close.loc[name] = dat['close'][close.columns] # 将指数收盘价数据添加到close 
            hfq_close.loc[name] = dat['close'][hfq_close.columns] # 将指数收盘价数据添加到hfq_close(指数无需复权)
        #更新数据
        pct_chg = pct_chg / 100
        self.close_file(pct_chg, 'pct_chg')
        self.close_file(close, 'close')
        self.close_file(hfq_close, 'hfq_close')
        #-------------------------------------------------------------
        #生成周期为1,3,6,12月收益率
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        tdays_be_month = self.trade_days_begin_end_of_month
        tdays_be_month = tdays_be_month[(tdays_be_month>=s)&(tdays_be_month<=e)].dropna(how='all')
        months_end = tdays_be_month.index
        hfq_close = self.hfq_close
        #***pct_chg_M 月频收益率指标（当月收益率）
        pct_chg_M = pd.DataFrame()
        for m_end_date in months_end:
            m_start_date = tdays_be_month.loc[m_end_date].values[0] # 获取当月首个交易日
            # 计算当月收益率：（月末收盘价 / 月初收盘价）-1
            pct_chg_M[self.month_map.loc[m_end_date]] = hfq_close[m_end_date] / hfq_close[m_start_date] - 1
        self.close_file(pct_chg_M, 'pct_chg_M')
        #pct_chg_Nm
        for period in (1,3,6,12):
            pct_chg_Nm = pd.DataFrame()
            if period != 1: 
                for m_end_date in months_end[::-1]:
                    try:
                        start_date_before_n_period = tdays_be_month.loc[self._get_date(m_end_date, -period+1, months_end)].values[0]
                        s = hfq_close[m_end_date] / hfq_close[start_date_before_n_period] - 1
                        pct_chg_Nm[self.month_map[m_end_date]] = s
                    except KeyError:
                        print(m_end_date)
                        break
            else:
                pct_chg_Nm = getattr(self, f'pct_chg_M', None)
            self.close_file(pct_chg_Nm, f"pctchg_{period}M")
            print(f'pct_chg_{period}M updated.')

    # 生成每日基本面指标
    def create_daily_basic_indicators(self):
        ''' 生成每日基本面指标
        Args:
            无显式参数，通过类属性获取数据
        Returns:
            无直接返回值，通过close_file方法存储生成指标

        实现功能：
        1. 创建换手率指标（单位转换：百分比→小数）
        2. 创建总市值指标（单位：亿元）
        实现步骤：
        - 从原始数据创建基础指标
        - 进行数据预处理（填充缺失值/过滤非交易日）
        - 存储预处理后的指标数据
        '''
        # 创建换手率指标
        self.create_indicator("__temp_daily_basic__", "turnover_rate", "turn")
        turn = self.turn / 100 # 单位转换：百分比→小数（例如：50% → 0.5）
        turn = self.preprocess(turn, suspend_days_process=True) # 预处理（处理停牌日数据，停牌日换手率设为0）
        self.close_file(turn, "turn")
        # 创建总市值指标（原始数据单位为万元）
        self.create_indicator("__temp_daily_basic__", "total_mv", "mkt_cap_ard")
        mkt_cap_ard = self.preprocess(self.mkt_cap_ard)
        self.close_file(mkt_cap_ard, "mkt_cap_ard")

    # 数据预处理
    def preprocess(self, datdf, suspend_days_process=False, val=np.nan):
        ''' 数据预处理
        Args:
            datdf (pd.DataFrame): 待处理数据框，索引为股票代码，列为日期
            suspend_days_process (bool): 是否处理停牌日数据（默认False）
            val (Any): 停牌日数据替代值（默认np.nan）
        Returns:
            pd.DataFrame: 处理后的数据框

        功能说明：
        1. 处理缺失值：沿时间轴向前/向后填充
        2. 过滤非上市日数据：将非上市期间的数值设为NaN
        3. 可选处理停牌日数据：将上市但停牌的数值设为指定值
        实现步骤：
        1. 创建数据副本避免污染原始数据
        2. 双向填充缺失值（先向前填充再向后填充）
        3. 获取股票上市状态矩阵
        4. 构建过滤条件并应用非上市日过滤
        5. 根据参数决定是否处理停牌日数据
        '''
        datdf = datdf.copy()
        # 沿列方向（时间轴）进行双向填充缺失值（先向前填充再向后填充）
        datdf = datdf.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        row_index, col_index = datdf.index, datdf.columns 
        # 获取股票上市状态矩阵（与当前数据维度对齐）
        liststatus = self.listday_matrix.loc[row_index, col_index]
        cond = (liststatus==1) # 构建过滤条件：仅保留上市日的有效数据
        datdf = datdf.where(cond) #将不是上市日的数值替换为nan
        if suspend_days_process: # 处理停牌日数据（当suspend_days_process为True时）
            # 获取交易状态矩阵（1表示正常交易，0表示停牌）
            tradestatus = self.trade_status.loc[row_index, col_index]
            cond = (liststatus==1) & (tradestatus==0)  # 构建复合条件：上市但停牌
            # ~cond表示"非停牌日或非上市日"，保留原值；停牌日设为val
            datdf = datdf.where(~cond, val) #将上市但停牌的数值设为指定值
        return datdf

    # 检查并清理空文件
    def clean_empty_files(self, dir_name):
        """检查指定目录下的文件是否为空,删除空文件
        Args:
            dir_name: 需要检查的目录名
        """
        tmp_dir = os.path.join(self.root, dir_name)
        for f in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, f)
            try:
                # 读取文件检查是否有数据
                df = pd.read_csv(file_path, encoding='gbk')
                if df.empty:
                    # 如果文件为空则删除
                    os.remove(file_path)
                    print(f"删除空文件: {f}")
            except Exception as e:
                print(f"处理文件 {f} 时发生错误: {str(e)}")
                # 如果文件损坏或无法读取也删除
                os.remove(file_path)
                print(f"删除问题文件: {f}")

    # 修改指定文件夹内所有文件的列名
    def rename_columns_in_folder(self, tmp_dir, column_mapping, file_extension='.csv', encoding='gbk'):
        """
        修改指定文件夹内所有文件的列名
        
        参数:
            folder_path (str): 文件夹路径
            column_mapping (dict): 列名映射字典，格式为 {'旧列名': '新列名', ...}
            file_extension (str): 文件扩展名，默认为'.csv'
            encoding (str): 文件编码，默认为'gbk'
        
        返回:
            dict: 包含处理结果的字典，格式为 {'成功': 成功文件数, '失败': 失败文件列表}
        
        示例:
            mapping = {'ts_code': 'stock_code', 'month_end': 'end_of_month'}
            result = rename_columns_in_folder('D:/MOE/data', mapping)
            print(f"成功处理 {result['成功']} 个文件")
            if result['失败']:
                print(f"处理失败的文件: {result['失败']}")
        """
        folder_path = os.path.join(self.root, tmp_dir)
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 '{folder_path}' 不存在")
            return {'成功': 0, '失败': []}
        
        # 获取文件夹中所有指定扩展名的文件
        files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
        
        if not files:
            print(f"警告: 文件夹 '{folder_path}' 中没有找到 {file_extension} 文件")
            return {'成功': 0, '失败': []}
        
        result = {'成功': 0, '失败': []}
        
        # 处理每个文件
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                # 读取文件
                df = pd.read_csv(file_path, encoding=encoding)
                
                # 检查文件中是否存在需要重命名的列
                columns_to_rename = {old: new for old, new in column_mapping.items() if old in df.columns}
                
                if not columns_to_rename:
                    print(f"跳过 {file_name}: 没有找到需要重命名的列")
                    continue
                
                # 重命名列
                df.rename(columns=columns_to_rename, inplace=True)
                
                # 保存修改后的文件
                df.to_csv(file_path, encoding=encoding, index=False)
                
                print(f"成功处理 {file_name}: 重命名了 {list(columns_to_rename.keys())} 列")
                result['成功'] += 1
                
            except Exception as e:
                print(f"处理 {file_name} 时出错: {str(e)}")
                result['失败'].append(file_name)
        
        return result


class TushareFetcher(RawDataFetcher):
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        # 从环境变量获取API密钥
        api_token = os.getenv('TUSHARE_API_TOKEN')
        if not api_token:
            raise ValueError("未找到TUSHARE_API_TOKEN环境变量，请在.env文件中设置")
        self.pro = ts.pro_api(api_token)
        super().__init__(using_fetch=True)

    # 全市场股票基础信息(self.meta)
    def fetch_meta_data(self):
        """获取全市场股票基础信息
        功能说明：
        1. 合并三种上市状态股票数据（正常上市D，终止上市，暂停上市P）
        2. 统一字段命名规范
        3. 数据去重和排序
        4. 最终数据存储（通过close_file方法）
        数据字段说明：
        - ts_code -> code: 股票代码
        - name -> sec_name: 证券简称
        - list_date -> ipo_date: 上市日期
        - delist_date: 退市日期
        """
        df_list = []
        # 获取正常上市股票数据 
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date')
        df_list.append(df)
        # 获取终止上市股票数据（list_status='D'）
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date', list_status='D')
        df_list.append(df)
        # 获取暂停上市股票数据（list_status='P'）
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date', list_status='P')
        df_list.append(df)
        df = pd.concat(df_list)
        df = df.rename(columns={"list_date":"ipo_date"}) # 上市日期
        df = df.rename(columns={'name':'sec_name'}) # 证券简称
        df = df.rename(columns={"ts_code":"code"}) # 股票代码
        df.drop_duplicates(subset=['code'], keep='first', inplace=True) # 去重处理（保留第一个出现的记录）
        df.sort_values(by=['ipo_date'], inplace=True) # 按上市日期排序
        #print(pd.to_datetime(df['ipo_date']))
        #df.reset_index(drop=True, inplace=True)
        df.set_index(['code'], inplace=True) # 设置股票代码为索引
        self.close_file(df, 'meta')

    # 交易日列表(self.tradedays)
    def fetch_trade_day(self):
        """ 交易日列表
        功能说明：
        1. 获取所有交易日列表
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - cal_date -> tradedays: 交易日日期
        - is_open -> is_open: 是否为交易日（1表示是交易日，0表示非交易日）
        """
        # 获取所有交易日列表
        df = self.pro.trade_cal(is_open='1')
        df = df[['cal_date','is_open']] # 只保留cal_date和is_open列
        df = df.rename(columns={"cal_date":"tradedays"}) # 交易日日期
        df.set_index(['tradedays'], inplace=True)
        self.close_file(df, 'tradedays')

    # 每月最后一个交易日和每月最后一个日历日的映射表(self.month_map)
    def fetch_month_map(self):
        """ 每月最后一个交易日和每月最后一个日历日的映射表
        功能说明：
        1. 获取每月最后一个交易日和每月最后一个日历日的映射表
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - trade_date -> trade_date: 交易日日期
        - calendar_date -> calendar_date: 日历日日期
        - trade_date -> calendar_date: 交易日到日历日的映射关系
        """
        tdays = self.tradedays
        s_dates = pd.Series(tdays, index=tdays)
        # 定义一个lambda函数，用于获取每个月的最后一个交易日
        func_last = lambda ser: ser.iat[-1]
        # 对交易日数据进行重采样，按月份进行分组，并应用lambda函数获取每个月的最后一个交易日
        new_dates = s_dates.resample('M').apply(func_last)
        month_map = new_dates.to_frame(name='trade_date')
        month_map.index.name = 'calendar_date'
        month_map.reset_index(inplace=True)
        month_map.set_index(['trade_date'], inplace=True)
        self.close_file(month_map, 'month_map')

    #------------------------------------------------------------------------------------
    #日数据
    # A股日线行情
    def daily(self, t):
        return self.pro.daily(trade_date=t)
    # 每日停复牌信息
    def suspend_d(self, t):
        return self.pro.suspend_d(trade_date=t)
    # 每日涨跌停信息
    def limit_list(self, t):
        return self.pro.limit_list(trade_date=t)
    # 复权因子
    def adj_factor(self, t):
        return self.pro.adj_factor(trade_date=t)
    # 每日指标
    def daily_basic(self, t):
        return self.pro.daily_basic(trade_date=t)
    # 个股资金流向
    def moneyflow(self, t):
        return self.pro.moneyflow(trade_date=t)
    # 中信行业成分
    def zx_industry(self, t):
        return self.pro.ci_index_member(trade_date=t)
    # 申万行业成分
    def sw_industry(self, t):
        return self.pro.index_member_all(trade_date=t)
    #------------------------------------------------------------------------------------
    def segment_op(limit, _max):
        """ 分段获取数据
        Args:
            limit (int): 每次获取的数据量
            _max (int): 最大获取的数据量
        Returns:
            function: 装饰器函数
        功能说明：
        1. 用于装饰需要分段获取数据的函数
        2. 按照limit大小分段遍历
        3. 设置数据偏移量
        4. 调用原始函数获取数据
        5. 拼接所有分段数据
        6. 返回最终结果
        """
        def segment_op_(f):
            # 装饰器函数,接收原始函数作为参数
            @wraps(f) 
            def wrapper(*args, **kwargs):
                # 用于存储所有分段数据的列表
                dfs = []
                # 按照limit大小分段遍历
                for i in range(0, _max, limit):
                    # 设置数据偏移量
                    kwargs['offset'] = i
                    # 调用原始函数获取数据
                    df = f(*args, **kwargs)
                    # 如果返回数据量小于limit,说明已经到最后一段
                    if len(df) < limit:
                        # 如果最后一段有数据则添加
                        if len(df) > 0:
                            dfs.append(df)
                        break
                    # 截取limit条数据
                    df = df.iloc[0:limit]
                    # 添加到结果列表
                    dfs.append(df)
                # 合并所有分段数据
                df = pd.concat(dfs, ignore_index=True)
                return df
            # 返回包装后的函数
            return wrapper
        # 返回装饰器函数
        return segment_op_
    #------------------------------------------------------------------------------------
    #季度数据
    @segment_op(limit=5000, _max=100000)
    def fina_indicator(self, *args, **kwargs):
        fields = '''ts_code,
        ann_date,
        end_date,
        eps,
        dt_eps,
        total_revenue_ps,
        revenue_ps,
        capital_rese_ps,
        surplus_rese_ps,
        undist_profit_ps,
        extra_item,
        profit_dedt,
        gross_margin,
        current_ratio,
        quick_ratio,
        cash_ratio,
        invturn_days,
        arturn_days,
        inv_turn,
        ar_turn,
        ca_turn,
        fa_turn,
        assets_turn,
        op_income,
        valuechange_income,
        interst_income,
        daa,
        ebit,
        ebitda,
        fcff,
        fcfe,
        current_exint,
        noncurrent_exint,
        interestdebt,
        netdebt,
        tangible_asset,
        working_capital,
        networking_capital,
        invest_capital,
        retained_earnings,
        diluted2_eps,
        bps,
        ocfps,
        retainedps,
        cfps,
        ebit_ps,
        fcff_ps,
        fcfe_ps,
        netprofit_margin,
        grossprofit_margin,
        cogs_of_sales,
        expense_of_sales,
        profit_to_gr,
        saleexp_to_gr,
        adminexp_of_gr,
        finaexp_of_gr,
        impai_ttm,
        gc_of_gr,
        op_of_gr,
        ebit_of_gr,
        roe,
        roe_waa,
        roe_dt,
        roa,
        npta,
        roic,
        roe_yearly,
        roa2_yearly,
        roe_avg,
        opincome_of_ebt,
        investincome_of_ebt,
        n_op_profit_of_ebt,
        tax_to_ebt,
        dtprofit_to_profit,
        salescash_to_or,
        ocf_to_or,
        ocf_to_opincome,
        capitalized_to_da,
        debt_to_assets,
        assets_to_eqt,
        dp_assets_to_eqt,
        ca_to_assets,
        nca_to_assets,
        tbassets_to_totalassets,
        int_to_talcap,
        eqt_to_talcapital,
        currentdebt_to_debt,
        longdeb_to_debt,
        ocf_to_shortdebt,
        debt_to_eqt,
        eqt_to_debt,
        eqt_to_interestdebt,
        tangibleasset_to_debt,
        tangasset_to_intdebt,
        tangibleasset_to_netdebt,
        ocf_to_debt,
        ocf_to_interestdebt,
        ocf_to_netdebt,
        ebit_to_interest,
        longdebt_to_workingcapital,
        ebitda_to_debt,
        turn_days,
        roa_yearly,
        roa_dp,
        fixed_assets,
        profit_prefin_exp,
        non_op_profit,
        op_to_ebt,
        nop_to_ebt,
        ocf_to_profit,
        cash_to_liqdebt,
        cash_to_liqdebt_withinterest,
        op_to_liqdebt,
        op_to_debt,
        roic_yearly,
        total_fa_trun,
        profit_to_op,
        q_opincome,
        q_investincome,
        q_dtprofit,
        q_eps,
        q_netprofit_margin,
        q_gsprofit_margin,
        q_exp_to_sales,
        q_profit_to_gr,
        q_saleexp_to_gr,
        q_adminexp_to_gr,
        q_finaexp_to_gr,
        q_impair_to_gr_ttm,
        q_gc_to_gr,
        q_op_to_gr,
        q_roe,
        q_dt_roe,
        q_npta,
        q_opincome_to_ebt,
        q_investincome_to_ebt,
        q_dtprofit_to_profit,
        q_salescash_to_or,
        q_ocf_to_sales,
        q_ocf_to_or,
        basic_eps_yoy,
        dt_eps_yoy,
        cfps_yoy,
        op_yoy,
        ebt_yoy,
        netprofit_yoy,
        dt_netprofit_yoy,
        ocf_yoy,
        roe_yoy,
        bps_yoy,
        assets_yoy,
        eqt_yoy,
        tr_yoy,
        or_yoy,
        q_gr_yoy,
        q_gr_qoq,
        q_sales_yoy,
        q_sales_qoq,
        q_op_yoy,
        q_op_qoq,
        q_profit_yoy,
        q_profit_qoq,
        q_netprofit_yoy,
        q_netprofit_qoq,
        equity_yoy,
        rd_exp,
        update_flag'''
        kwargs['fields'] = fields
        return self.pro.fina_indicator_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def income(self, *args, **kwargs):
        return self.pro.income_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def balancesheet(self, *args, **kwargs):
        return self.pro.balancesheet_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def cashflow(self, *args, **kwargs):
        return self.pro.cashflow_vip(*args, **kwargs)

    #------------------------------------------------------------------------------------
    #指数日行情
    def index_daily(self):
        ''' 指数日行情
        功能说明：
        1. 获取指数日行情数据
        2. 数据存储（通过close_file方法）
        '''
        index_list = ['000001.SH', '000300.SH', '000905.SH']
        tmp_dir = os.path.join(self.root, "__temp_index_daily__")
        os.makedirs(tmp_dir, exist_ok=True)
        for i in index_list:
            try:
                # 获取指数日行情数据
                df = self.pro.index_daily(ts_code=i)
                path = os.path.join(tmp_dir, i+".csv")
                df.to_csv(path, encoding='gbk')
                print(i+".csv 写入成功！")
            except Exception as e:
                print(f"获取 {i} 数据失败: {str(e)}")

    # 000300.SH指数成分股所占权重
    def hs300_wt(self, t):
        return self.pro.index_weight(index_code='000300.SH', trade_date=t)

    # 000905.SH指数成分股所占权重
    def zz500_wt(self, t):
        return self.pro.index_weight(index_code='000905.SH', trade_date=t)
    #------------------------------------------------------------------------------------
        '''
        通过上面的函数,会从tushare把原始数据下载并保存到本地raw_data目录中
        raw_data/src目录: 股票基础列表,成交日列表
        raw_data/__temp_adj_factor__目录: 复权因子表(日频数据)
        raw_data/__temp_daily__目录: 每日行情表(日频数据)
        raw_data/__temp_daily_basic__目录: 每日指标表(日频数据)
        raw_data/__temp_limit_list__目录: 每日涨跌停表(日频数据)
        raw_data/__temp_moneyflow__目录: 每日个股资金流向表(日频数据)
        raw_data/__temp_suspend_d__目录: 每日停复牌表(日频数据)
        raw_data/__temp_index_daily__目录: 每日指数行情(日频数据)
        raw_data/__temp_balancesheet__目录: 资产负债表(季频数据)
        raw_data/__temp_cashflow__目录: 现金流量表(季频数据)
        raw_data/__temp_fina_indicator__目录: 财务指标表(季频数据)
        raw_data/__temp_income__目录: 利润表(季频数据)
        
        下面开始的函数主要就是通过上面这些原始数据生成一些月频基础指标,主要有三种形式:
        1. 通过 <日频数据> 生成 <月频指标>
        2. 通过 <季频数据> 生成 <月频指标>
        3. 通过 <日频数据>和<季频数据> 混合生成 <月频指标>
        '''

    # 股票上市存续周期日矩阵(self.listday_matrix)
    def create_listday_matrix(self):
        ''' 股票上市存续周期日矩阵
        功能说明：
        1. 计算每个股票在每个交易日的上市状态
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - listday -> listday: 上市状态（1表示上市，0表示未上市）
        '''
        all_stocks_info = self.meta # 股票基础信息
        trade_days = self.tradedays # 交易日列表
        # 定义一个函数,用于判断股票是否在某个交易日上市
        def if_listed(series):
            nonlocal all_stocks_info
            code = series.name
            ipo_date = all_stocks_info.at[code, 'ipo_date'] # 上市日期
            delist_date = all_stocks_info.at[code, 'delist_date'] # 退市日期
            daterange = series.index # 交易日日期范围
            # 如果退市日期为NaT,则表示股票未退市,只判断交易日是否大于等于上市日期
            if delist_date is pd.NaT:
                res = np.where(daterange >= ipo_date, 1, 0)
            # 如果退市日期不为NaT,则表示股票已退市,需要判断交易日是否在上市日期和退市日期之间
            else:
                res = np.where(daterange < ipo_date, 0, np.where(daterange <= delist_date, 1, 0))
            return pd.Series(res, index=series.index)
        # 计算每个股票在每个交易日的上市状态
        listday_dat = pd.DataFrame(index=all_stocks_info.index, columns=trade_days)
        listday_dat = listday_dat.apply(if_listed, axis=1) # 应用函数
        self.close_file(listday_dat, 'listday_matrix')

    # 每月第一个和最后一个交易日映射(self.trade_days_begin_end_of_month)
    def create_month_tdays_begin_end(self, latest_month_end_tradeday=None):
        ''' 每月第一个和最后一个交易日映射
        Args:
            latest_month_end_tradeday (str, optional): 最新的月最后交易日. Defaults to None.
        功能说明：
        1. 计算每月第一个和最后一个交易日的映射关系
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - month_start -> month_start: 每月第一个交易日
        - month_end -> month_end: 每月最后一个交易日
        '''
        tdays = self.tradedays # 交易日列表
        # 计算每月第一个和最后一个交易日的映射关系
        months_end = tdays[0:1] + list(after_d for before_d, after_d in zip(tdays[:-1], tdays[1:]) if before_d.month != after_d.month)
        months_start = list(before_d for before_d, after_d in zip(tdays[:-1], tdays[1:]) if before_d.month != after_d.month) + tdays[-1:]
        # 处理最新的月最后交易日
        if latest_month_end_tradeday is None:
            latest_month_end_tradeday = self.month_map.index[-1]
        # 处理最新的月最后交易日在交易日列表中的情况
        if months_end[-1] > latest_month_end_tradeday:
            months_start, months_end = months_start[:-1], months_end[:-1]
        trade_days_be_month = pd.DataFrame(months_end, index=months_start, columns=['month_end'])
        trade_days_be_month.index.name = 'month_start'
        self.close_file(trade_days_be_month, 'trade_days_begin_end_of_month')

    # 股票停复牌状态(self.trade_status)
    def create_trade_status(self):
        ''' 股票停复牌状态
        功能说明：
        1. 计算每个股票在每个交易日的停复牌状态
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - trade_status -> trade_status: 停复牌状态（0表示停牌，1表示正常）
        '''
        tmp_dir = os.path.join(self.root, "__temp_suspend_d__")
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays) # 交易日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        df.loc[:, :] = 1 #默认都是正常状态
        # 遍历每个交易日,读取停牌数据,并更新df
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col='ts_code', engine='python', encoding='gbk')
            #==================================================
            #有些股票已经变更名字和证劵代码,需要修改
            index = dat.index.to_series()
            index = index.replace("000022.SZ", "001872.SZ")
            index = index.replace("601313.SH", "601360.SH")
            index = index.replace("000043.SZ", "001914.SZ")
            index = index.replace("300114.SZ", "302132.SZ")
            index = index.replace("830799.BJ", "920799.BJ")
            index = index.replace("834682.BJ", "920682.BJ")
            index = index.replace("833819.BJ", "920819.BJ")
            index = index.replace("831445.BJ", "920445.BJ")
            index = index.replace("839167.BJ", "920167.BJ")
            index = index.replace("430489.BJ", "920489.BJ")
            #==================================================
            df.loc[index, tday] = 0 #停牌的设置为0
            print(tday)
        self.close_file(df, "trade_status")

    # 股票涨跌停状态(self.maxupordown)
    def create_maxupordown(self):
        ''' 股票涨跌停状态
        功能说明：
        1. 计算每个股票在每个交易日的涨跌停状态
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - maxupordown -> maxupordown: 涨跌停状态（0表示涨跌停，1表示正常）
        '''
        tmp_dir = os.path.join(self.root, "__temp_limit_list__")
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays) # 交易日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        df.loc[:, :] = 0 #默认都没有涨跌停
        # 遍历每个交易日,读取涨跌停数据,并更新df
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col='ts_code', engine='python', encoding='gbk')
            #==================================================
            #有些股票已经变更名字和证劵代码,需要修改
            index = dat.index.to_series()
            index = index.replace("000022.SZ", "001872.SZ")
            index = index.replace("601313.SH", "601360.SH")
            index = index.replace("000043.SZ", "001914.SZ")
            index = index.replace("300114.SZ", "302132.SZ")
            index = index.replace("830799.BJ", "920799.BJ")
            index = index.replace("834682.BJ", "920682.BJ")
            index = index.replace("833819.BJ", "920819.BJ")
            index = index.replace("831445.BJ", "920445.BJ")
            index = index.replace("839167.BJ", "920167.BJ")
            index = index.replace("430489.BJ", "920489.BJ")
            #==================================================
            df.loc[index, tday] = 1 #涨跌停的设置为1
            print(tday)
        self.close_file(df, "maxupordown")

    # 日换手率(self.turn)
    def create_turn_d(self):
        ''' 日换手率
        功能说明：
        1. 计算每个股票在每个交易日的换手率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - turn -> turn: 换手率
        '''
        self.create_indicator("__temp_daily_basic__", "turnover_rate", "turn")
        turn = self.turn / 100 # 单位转换：百分比→小数（例如：50% → 0.5）
        turn = self.preprocess(turn, suspend_days_process=True) # 处理停牌数据
        self.close_file(turn, "turn")

    # 月流通市值(self.mkt_cap_float_m)
    def create_mkt_cap_float_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的流通市值
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - mkt_cap_float_m -> mkt_cap_float_m: 流通市值
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') # 起始日期
        e = pd.to_datetime('20250430') # 结束日期
        new_tdays = self._get_trade_days(s, e, "M") # 月最后一个交易日列表
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] # 月最后一个日历日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] # 月最后一个日历日
            df[caldate] = dat["circ_mv"] # 流通市值
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "mkt_cap_float_m")

    # 月市盈率(self.pe_ttm_m)
    def create_pe_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的市盈率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - pe_ttm_m -> pe_ttm_m: 市盈率
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') # 起始日期
        e = pd.to_datetime('20250430') # 结束日期
        new_tdays = self._get_trade_days(s, e, "M") # 月最后一个交易日列表
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] # 月最后一个日历日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] # 月最后一个日历日
            df[caldate] = dat["pe_ttm"] # 市盈率
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "pe_ttm_m")

    # 月市净率(self.val_pe_deducted_ttm_m)
    def create_val_pe_deducted_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的市净率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - val_pe_deducted_ttm_m -> val_pe_deducted_ttm_m: 市净率
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') # 起始日期
        e = pd.to_datetime('20250430') # 结束日期
        new_tdays = self._get_trade_days(s, e, "M") # 月最后一个交易日列表
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] # 月最后一个日历日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] # 月最后一个日历日
            df[caldate] = dat["pe"] #临时先用pe替代 市净率
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "val_pe_deducted_ttm_m")

    # 月市销率(self.pb_lf_m)
    def create_pb_lf_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的市销率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - pb_lf_m -> pb_lf_m: 市销率
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') # 起始日期
        e = pd.to_datetime('20250430') # 结束日期
        new_tdays = self._get_trade_days(s, e, "M") # 月最后一个交易日列表
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta # 股票基础信息
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] # 月最后一个日历日
            df[caldate] = dat["pb"] #市销率
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "pb_lf_m")

    # 月市现率(self.ps_ttm_m)
    def create_ps_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的市现率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - ps_ttm_m -> ps_ttm_m: 市现率
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] # 月最后一个日历日列表
        all_stocks_info = self.meta # 股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] # 月最后一个日历日
            df[caldate] = dat["ps_ttm"] #市现率
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "ps_ttm_m")

    # 股票在指定交易日期的收盘价 / 现金及现金等价物净增加额TTM(self.pcf_ncf_ttm_m)
    def create_pcf_ncf_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的现金及现金等价物净增加额TTM
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - pcf_ncf_ttm_m -> pcf_ncf_ttm_m: 现金及现金等价物净增加额TTM
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta #股票基础信息
        #-------------------------------------------------------
        #总市值指标(月频)
        df_total_mv = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays) #总市值指标(月频)
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__") #每日指标表
        #遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] #月最后一个日历日
            df_total_mv[caldate] = dat["total_mv"] #总市值
            print(caldate)
        #df_total_mv = df_total_mv.dropna(how='all') #删掉全为空的一行
        print(df_total_mv.head()) #总市值指标ok
        print('总市值指标ok')
        #-------------------------------------------------------
        #现金增加额指标(季频)
        tmp_dir = os.path.join(self.root, "__temp_cashflow__") #现金流量表
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        df_cfps = pd.DataFrame(index=all_stocks_info.index, columns=qdays) #现金增加额指标(季频)
        df_ann_date = pd.DataFrame(index=all_stocks_info.index, columns=qdays) #财报发布日期(季频)
        #遍历每个季度最后一天,读取季频数据,并更新df
        for qday in qdays:
            name = qday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date'])
            diff = dat.index.difference(df_cfps.index) #删除没在股票基础列表中多余的股票行
            dat = dat.loc[~dat.index.isin(diff)] #方法1
            #dat = dat.drop(labels=diff) #方法2
            #
            #x = dat.index.to_series()
            #print(x)
            #x = x.groupby(['ts_code'])
            #print(x)
            #print(x.count())
            #print(x.count()>1)
            #print(dat[x.count()>1])
            #
            #x = dat.index
            #print(x.duplicated())
            #print(dat[x.duplicated()])
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            df_cfps[qday] = dat["n_incr_cash_cash_equ"] #现金及现金等价物净增加额
            df_ann_date[qday] = dat["ann_date"] #财报发布日期
            print(qday)
        #df_cfps = df_cfps.dropna(how='all') #删掉全为空的一行
        print(df_cfps) #现金增加额指标ok
        print('现金增加额指标ok')
        #-------------------------------------------------------
        # #现金增加额指标可能有空值,利用线性插值补全(这步可以不做)
        # df_cfps_t = df_cfps.T #把时间变成索引,股票变成列名
        # df_cfps_t.index = pd.to_datetime(df_cfps_t.index)  # 确保索引是DatetimeIndex
        # # 定义一个函数,用于处理每只股票的每一年的数据
        # def _w(ser):
        #     # 一年内如果第四季度(年报)指标值为空,那么整年四个季度都设置为空
        #     if len(ser) < 4:
        #         return
        #     if pd.isna(ser.iloc[3]):
        #         ser.iloc[:] = np.nan
        #         df_cfps_t.loc[ser.index, ser.name] = ser  # 回填
        #         return
        #     # 检查是否有缺失值需要插值处理
        #     if ser.isna().any():
        #         # 第一季度必须保证有值才能进行插值
        #         if pd.isna(ser.iloc[0]):
        #             ser.iloc[0] = ser.iloc[3] / 4 # 第一季度如果为空,就用全年的均值进行填充
        #         # 使用线性插值填充缺失值
        #         ser = ser.interpolate()
        #     # 回填处理后的数据
        #     df_cfps_t.loc[ser.index, ser.name] = ser
        # # 修改resample方式，确保填充完整年度数据
        # df_cfps_t = df_cfps_t.resample('Q').asfreq().resample('A').apply(_w)
        # df_cfps = df_cfps_t.T #变回来:股票为索引,日期为列名
        #-------------------------------------------------------
        #计算结果指标(月频)
        df_result = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        '''
        算法:
        (1)最新报告期是年报，则TTM=年报；
        (2)最新报告期不是年报，则TTM=本期+(上年年报-上年同期)，如果本期、上年年报、上年同期存在空值，则不计算，返回空值；
        (3)最新报告期通过财报发布时间进行判断,防止前视偏差。
        '''
        #按时间和股票逐个开始计算
        for calday in df_result.columns: #每月最后一天
            for stock in df_result.index:
                tmap = df_ann_date.loc[stock] #tmap索引为报告期(每季度最后一天),值为相应财报发布时间
                tmap = tmap[tmap<calday] #在那个历史节点,只能使用已经发布的财报,防止使用未来数据
                try:
                    d = tmap.index[-1] #已经发布的财报里面最近一期的时间(某季度最后一天)
                    if d.quarter == 4: #最近一期财报是年报(第4季度)
                        ttm_value = df_cfps.loc[stock, d]
                    else: #最近一期财报是1季度,2季度,或者3季度的情形
                        last_q_4 = tmap.index[-1-d.quarter] #相对于那一个历史节点的上一年年报的时间
                        last_q_same = tmap.index[-1-4] #相对于那一个历史节点的上一年同期的时间
                        ttm_value = df_cfps.loc[stock, d] + (df_cfps.loc[stock, last_q_4] - df_cfps.loc[stock, last_q_same]) #TTM=本期+(上年年报-上年同期)
                    #总市值/现金及现金等价物净增加额(TTM)
                    df_result.loc[stock, calday] = df_total_mv.loc[stock, calday]/ttm_value
                except:
                    pass
        df_result = df_result.dropna(how='all') #删掉全为空的一行
        self.close_file(df_result, "pcf_ncf_ttm_m")

    # 股票在指定交易日期的收盘价 / 营业收入TTM(self.pcf_ocf_ttm_m)
    def create_pcf_ocf_ttm_m(self):
        '''通过日频数据创建月频指标(可统一为单个函数)
        本函数与上面的create_pcf_ncf_ttm_m类似,逻辑更优化
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的营业收入TTM
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - pcf_ocf_ttm_m -> pcf_ocf_ttm_m: 营业收入TTM
        '''
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta #股票基础信息
        #-------------------------------------------------------
        #总市值指标(月频)
        df_total_mv = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays) #总市值指标(月频)
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__") #每日指标表
        for tday in new_tdays: #每月最后一个交易日
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] #每月最后一个日历日
            df_total_mv[caldate] = dat["total_mv"] #总市值
            print(caldate)
        df_total_mv = df_total_mv.dropna(how='all') #删掉全为空的一行
        #-------------------------------------------------------
        tmp_dir = os.path.join(self.root, "__temp_cashflow__") #现金流量表
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        panel = {}
        #遍历每个季度最后一天,读取季频数据,并更新df
        for d in qdays:
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.loc[~dat.index.isin(diff)] 
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0'] #删除多余的列
            panel[d] = dat
            print(d)
        panel = pd.concat(panel, axis=0)
        panel = panel.stack().unstack(level=(1, -1))
        #-------------------------------------------------------
        #开始计算结果指标(月频)
        df_result = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        '''
        算法:
        (1)最新报告期是年报，则TTM=年报；
        (2)最新报告期不是年报，则TTM=本期+(上年年报-上年同期)，如果本期、上年年报、上年同期存在空值，则不计算，返回空值；
        (3)最新报告期通过财报发布时间进行判断,防止前视偏差。
        '''
        #按时间和股票逐个开始计算
        for calday in df_result.columns: #每月最后一天
            for stock in df_result.index: #每只股票
                try:
                    datdf = panel[stock] #股票的所有财报数据
                    datdf = datdf.loc[datdf['ann_date']<calday] #在那个历史节点,只能使用已经发布的财报,防止使用未来数据
                    d = datdf.iloc[-1].name #已经发布的财报里面最近一期的时间(某季度最后一天)
                    if d.quarter == 4: #最近一期财报是年报(第4季度)
                        ttm_value = datdf.iloc[-1].at['n_cashflow_act']
                    else: #最近一期财报是1季度,2季度,或者3季度的情形
                        last_q_4 = datdf.iloc[-1-d.quarter] #相对于那一个历史节点的上一年年报
                        last_q_same = datdf.iloc[-1-4] #相对于那一个历史节点的上一年同期
                        #TTM=本期+(上年年报-上年同期)
                        ttm_value = datdf.iloc[-1].at['n_cashflow_act'] + (last_q_4.at['n_cashflow_act'] - last_q_same.at['n_cashflow_act'])
                    #总市值/经营活动产生的现金流量净额(TTM)
                    df_result.at[stock, calday] = df_total_mv.at[stock, calday]/ttm_value
                except:
                    pass
            print(calday)
        df_result = df_result.dropna(how='all') #删掉全为空的一行
        self.close_file(df_result, "pcf_ocf_ttm_m")

    # 月复利率(self.dividendyield2_m)
    def create_dividendyield2_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        功能说明：
        1. 计算每个股票在每个月最后一个交易日的月复利率
        2. 数据存储（通过close_file方法）
        数据字段说明：
        - code -> code: 股票代码
        - tradedays -> tradedays: 交易日日期
        - dividendyield2_m -> dividendyield2_m: 月复利率
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta #股票基础信息
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        # 遍历每个月最后一个交易日,读取日频数据,并更新df
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["dv_ttm"] #月复利率
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "dividendyield2_m")

    # 股票在指定交易日期的收盘价 / 净利润TTM(self.profit_ttm_G_m)
    def create_profit_ttm_G_m(self):
        ''' 通过季频数据创建月频指标,可以直接用create_indicator_m_by_q代替
        '''
        s = pd.to_datetime('20140101') #统计周期开始
        e = pd.to_datetime('20250430') #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天
        all_stocks_info = self.meta #股票基础信息
        tmp_dir = os.path.join(self.root, "__temp_fina_indicator__") #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        panel = pd.concat(panel, axis=0)
        panel = panel.stack().unstack(level=(1, -1))
        #开始计算结果指标(月频)
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
            for stock in df.index: #每只股票
                try:
                    datdf = panel[stock] #股票的所有财报数据
                    datdf = datdf.loc[datdf['ann_date']<d] #在那个历史节点,只能使用已经发布的财报,防止使用未来数据
                    df.at[stock, d] = datdf.iloc[-1].at['q_profit_yoy'] #最近一期财报的同比增长
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "profit_ttm_G_m")


if __name__ == '__main__':
    fetcher = TushareFetcher()
    # print("开始获取元数据...")
    # fetcher.fetch_meta_data()
    # print("元数据获取完成")
    # print("开始获取交易日数据...")
    # fetcher.fetch_trade_day()
    # print("交易日数据获取完成")
    # print("开始获取月份映射表...")
    # fetcher.fetch_month_map()
    # print("月份映射表获取完成")
    # print("开始获取日行情数据...")
    # fetcher.ensure_data(fetcher.daily, "__temp_daily__") #日行情表
    # print("日行情数据获取完成")
    # print("开始获取停牌数据...")
    # fetcher.ensure_data(fetcher.suspend_d, "__temp_suspend_d__") #停牌表
    # print("停牌数据获取完成")
    # print("开始获取涨跌停数据...")
    # fetcher.ensure_data(fetcher.limit_list, "__temp_limit_list__") #涨跌停表
    # print("涨跌停数据获取完成")
    # print("开始获取复权因子数据...")
    # fetcher.ensure_data(fetcher.adj_factor, "__temp_adj_factor__") #复权因子表
    # print("复权因子数据获取完成")
    # print("开始获取每日指标数据...")
    # fetcher.ensure_data(fetcher.daily_basic, "__temp_daily_basic__") #每日指标表
    # print("每日指标数据获取完成")
    # print("开始获取资金流数据...")
    # fetcher.ensure_data(fetcher.moneyflow, "__temp_moneyflow__") #资金流表
    # print("资金流数据获取完成")
    # print("开始获取中信行业成分数据...")
    # fetcher.ensure_data(fetcher.zx_industry, "__zx_industry__") #行业分类表
    # print("中信行业成分数据获取完成")
    # print("开始获取申万行业成分数据...")
    # fetcher.ensure_data(fetcher.sw_industry, "__sw_industry__") #行业分类表
    # print("申万行业成分数据获取完成")
    # print("开始获取000300.SH指数成分股所占权重数据...")
    # fetcher.ensure_data(fetcher.hs300_wt, "__hs300_wt__") #000300.SH指数成分股所占权重
    # print("000300.SH指数成分股所占权重数据获取完成")
    # print("开始获取000905.SH指数成分股所占权重数据...")
    # fetcher.ensure_data(fetcher.zz500_wt, "__zz500_wt__") #000905.SH指数成分股所占权重
    # print("000905.SH指数成分股所占权重数据获取完成")
    # print("开始获取财务指标数据...")
    # fetcher.ensure_data_by_q(fetcher.fina_indicator, "__temp_fina_indicator__") #财务指标表
    # print("财务指标数据获取完成")
    # print("开始获取利润表数据...")
    # fetcher.ensure_data_by_q(fetcher.income, "__temp_income__") #利润表
    # print("利润表数据获取完成")
    # print("开始获取资产负债表数据...")
    # fetcher.ensure_data_by_q(fetcher.balancesheet, "__temp_balancesheet__") #资产负债表
    # print("资产负债表数据获取完成")
    # print("开始获取现金流表数据...")
    # fetcher.ensure_data_by_q(fetcher.cashflow, "__temp_cashflow__") #现金流表
    # print("现金流表数据获取完成")
    # print("开始获取指数日线数据...")
    # fetcher.index_daily()
    # print("指数日线数据获取完成")
    # print("开始生成上市日矩阵...")
    # fetcher.create_listday_matrix()
    # print("上市日矩阵生成完成")
    # print("开始创建月份交易日...")
    # fetcher.create_month_tdays_begin_end()
    # print("月份交易日创建完成")
    # print("开始生成交易状态数据...")
    # fetcher.create_trade_status()  #交易状态数据
    # print("交易状态数据生成完成")
    # print("开始获取涨跌停数据...")
    # fetcher.create_maxupordown()
    # print("涨跌停数据获取完成")
    # print("开始生成日频换手率...")
    # fetcher.create_turn_d()
    # print("日频换手率生成完成")
    # print("开始创建复权因子...")
    # fetcher.create_indicator("__temp_adj_factor__", "adj_factor", "adjfactor")
    # print("复权因子创建完成")
    # print("开始生成流通市值数据...")
    # fetcher.create_mkt_cap_float_m()
    # print("流通市值数据生成完成")
    # print("开始计算市盈率(TTM)...")
    # fetcher.create_pe_ttm_m()
    # print("市盈率(TTM)计算完成")
    # print("开始创建扣除TTM估值数据...")
    # fetcher.create_val_pe_deducted_ttm_m()
    # print("扣除TTM估值数据创建完成")
    # print("开始生成市净率(LF)...")
    # fetcher.create_pb_lf_m()
    # print("市净率(LF)生成完成")
    # print("开始计算市销率(TTM)...")
    # fetcher.create_ps_ttm_m()
    # print("市销率(TTM)计算完成")
    # print("开始获取现金流折现估值...")
    # fetcher.create_pcf_ncf_ttm_m()
    # print("现金流折现估值获取完成")
    # print("开始生成经营现金流估值...")
    # fetcher.create_pcf_ocf_ttm_m()
    # print("经营现金流估值生成完成")
    # print("开始计算股息率2指标...")
    # fetcher.create_dividendyield2_m()
    # print("股息率2指标计算完成")
    # print("开始创建TTM利润增长率...")
    # fetcher.create_profit_ttm_G_m()
    # print("TTM利润增长率创建完成")
    # print("开始生成中信行业分类数据...")
    # fetcher.create_indicator_m_by_d("__zx_industry__", "l1_name", "industry_citic")
    # print("中信行业分类数据生成完成")
    # print("开始生成申万行业分类数据...")
    # fetcher.create_indicator_m_by_d("__sw_industry__", "l1_name", "industry_sw")
    # print("申万行业分类数据生成完成")
    # print("开始生成000300指数成分股权重数据...")
    # fetcher.create_indicator("__hs300_wt__", "weight", "hs300_wt")
    # print("000300指数成分股权重数据生成完成")
    # print("开始生成000905指数成分股权重数据...")
    # fetcher.create_indicator("__zz500_wt__", "weight", "zz500_wt")
    # print("000905指数成分股权重数据生成完成")
    # print("开始处理季度销售额同比增长率...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_sales_yoy", "qfa_yoysales_m")
    # print("季度销售额同比增长率处理完成")
    # print("开始生成季度利润同比增长指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_profit_yoy", "qfa_yoyprofit_m")
    # print("季度利润同比增长指标生成完成")
    # print("正在创建经营现金流增长率...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "ocf_yoy", "qfa_yoyocf_m")
    # print("经营现金流增长率创建完成")
    # print("开始计算ROE增长率...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yoy", "qfa_roe_G_m")
    # print("ROE增长率计算完成")
    # print("开始生成季度ROE指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_roe", "qfa_roe_m")
    # print("季度ROE指标生成完成")
    # print("正在计算年化ROE指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yearly", "roe_ttm2_m")
    # print("年化ROE指标计算完成")
    # print("开始生成ROA指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa", "qfa_roa_m")
    # print("ROA指标生成完成")
    # print("正在计算年化ROA指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa_yearly", "roa2_ttm2_m")
    # print("年化ROA指标计算完成")
    # print("开始生成毛利率指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_gsprofit_margin", "qfa_grossprofitmargin_m")
    # print("毛利率指标生成完成")
    # print("开始生成毛利率TTM指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "grossprofit_margin", "grossprofitmargin_ttm2_m") #毛利率TTM指标
    # print("毛利率TTM指标生成完成")
    # print("开始创建资产周转率指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_turn", "turnover_ttm_m") #资产周转率
    # print("资产周转率指标创建完成")
    # print("正在生成资产权益比...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_to_eqt", "assetstoequity_m")
    # print("资产权益比生成完成")
    # print("开始计算长期负债权益比...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "debt_to_eqt", "longdebttoequity_m") #长期负债权益比
    # print("长期负债权益比计算完成")
    # print("正在处理现金流动负债比...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "cash_to_liqdebt", "cashtocurrentdebt_m")
    # print("现金流动负债比处理完成")
    # print("开始生成流动比率指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "current_ratio", "current_m")
    # print("流动比率指标生成完成")
    # print("开始生成每日行情指标...")
    # fetcher.create_daily_quote_indicators()
    # print("每日行情指标生成完成!")
    # print("正在生成流通市值指标...")
    # fetcher.create_indicator("__temp_daily_basic__", "circ_mv", "mkt_cap_float")
    # print("流通市值指标生成完成")
    # print("正在生成总市值指标...")
    # fetcher.create_indicator("__temp_daily_basic__", "total_mv", "mkt_cap_ard")
    # print("总市值指标生成完成")
    # print("正在生成长期债务占比指标...")
    # fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "longdeb_to_debt", "longdebttodebt_lyr_m")
    # print("长期债务占比指标生成完成")
    # print("正在生成总负债指标...")
    # fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_liab", "tot_liab_lyr_m")
    # print("总负债指标生成完成")
    # print("正在生成其他权益工具指标...")
    # fetcher.create_indicator_m_by_q("__temp_balancesheet__", "oth_eqt_tools_p_shr", "other_equity_instruments_PRE_lyr_m")
    # print("其他权益工具指标生成完成")
    # print("正在生成股东权益合计指标...")
    # fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_hldr_eqy_inc_min_int", "tot_equity_lyr_m")
    # print("股东权益合计指标生成完成")
    # print("正在生成总资产指标...")
    # fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_assets", "tot_assets_lyr_m")
    # print("总资产指标生成完成")
    #
