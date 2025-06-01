import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
from datetime import datetime
from functools import reduce
from itertools import dropwhile
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

WORK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
if not os.path.exists(WORK_PATH): # 如果路径不存在，则创建路径
    os.makedirs(WORK_PATH)

class FileAlreadyExistError(Exception): 
    pass

class lazyproperty:
    """延迟属性装饰器类
    用于实现属性的延迟加载，只在首次访问时计算属性值
    """
    def __init__(self, func):
        """
        初始化装饰器
        Args:
            func: 被装饰的方法
        """
        self.func = func

    def __get__(self, instance, cls):
        """
        描述符的__get__方法
        Args:
            instance: 实例对象
            cls: 类对象
        Returns:
            如果通过类访问，返回装饰器本身
            如果通过实例访问，计算并缓存属性值，然后返回该值
        """
        if instance is None:
            return self
        else:
            value = self.func(instance)
            # 将计算得到的属性值存储在实例的属性字典中
            setattr(instance, self.func.__name__, value)
            return value

class Data:
    # 数据起止日期和频率设置
    startday = "20140101"
    endday = "20250430"
    #endday = pd.tseries.offsets.datetime.now().strftime("%Y%m%d")
    freq = "M"

    # 数据文件路径和文件名设置
    root = WORK_PATH
    metafile = 'all_stocks.xlsx'
    mmapfile = 'month_map.xlsx'
    month_group_file = 'month_group.xlsx'
    tradedays_file = 'tradedays.xlsx'
    tdays_be_m_file = 'trade_days_begin_end_of_month.xlsx'

    # 价值类因子原始指标和目标指标
    value_indicators = [
            'pe_ttm', 'val_pe_deducted_ttm', 'pb_lf', 'ps_ttm',
            'pcf_ncf_ttm', 'pcf_ocf_ttm', 'dividendyield2', 'profit_ttm'
            ]
    value_target_indicators = [
            "EP", "EPcut", "BP", "SP",
            "NCFP", "OCFP", "DP", "G/PE"
            ]

    # 成长类因子原始指标和目标指标
    growth_indicators = [
            "qfa_yoysales", "qfa_yoyprofit", "qfa_yoyocf", "qfa_roe_G_m"
            ]
    growth_target_indicators = [
            "Sales_G_q", "Profit_G_q", "OCF_G_q", "ROE_G_q"
            ]

    # 财务类因子原始指标和目标指标
    finance_indicators = [
            "roe_ttm2_m", "qfa_roe_m",
            "roa2_ttm2_m", "qfa_roa_m",
            "grossprofitmargin_ttm2_m", "qfa_grossprofitmargin_m",
            "deductedprofit_ttm", "qfa_deductedprofit_m", "or_ttm", "qfa_oper_rev_m",
            "turnover_ttm_m", "qfa_netprofitmargin_m",
            "ocfps_ttm", "eps_ttm", "qfa_net_profit_is_m", "qfa_net_cash_flows_oper_act_m"
            ]
    finance_target_indicators = [
            "ROE_q", "ROE_ttm",
            "ROA_q", "ROA_ttm",
            "grossprofitmargin_q", "grossprofitmargin_ttm",
            "profitmargin_q", "profitmargin_ttm",
            "assetturnover_q", "assetturnover_ttm",
            "operationcashflowratio_q", "operationcashflowratio_ttm"
            ]

    # 杠杆类因子原始指标和目标指标
    leverage_indicators = [
            "assetstoequity_m", "longdebttoequity_m",
            "cashtocurrentdebt_m", "current_m"
            ]
    leverage_target_indicators = [
            "financial_leverage", "debtequityratio",
            "cashratio", "currentratio"
            ]

    # 市场类因子原始指标和目标指标
    cal_indicators = ["mkt_cap_float", "holder_avgpct", "holder_num"]
    cal_target_indicators = [
            "ln_capital",
            "HAlpha", "return_1m", "return_3m", "return_6m", "return_12m",
            "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
            "exp_wgt_return_1m",  "exp_wgt_return_3m",  "exp_wgt_return_6m", "exp_wgt_return_12m",
            "std_1m", "std_3m", "std_6m", "std_12m",
            "beta",
            "turn_1m", "turn_3m", "turn_6m", "turn_12m",
            "bias_turn_1m", "bias_turn_3m", "bias_turn_6m", "bias_turn_12m",
            "holder_avgpctchange",
            ]

    # 技术类因子原始指标和目标指标
    tech_indicators = [
            "MACD", "RSI", "PSY", "BIAS"
            ]
    tech_target_indicators = [
            "MACD", "DEA", "DIF", "RSI", "PSY", "BIAS"
            ]

    # Barra行情类因子原始指标和目标指标
    barra_quote_indicators = [
            "mkt_cap_float", "pct_chg", "amt"
            ]
    barra_quote_target_indicators = [
            "LNCAP_barra", "MIDCAP_barra",
            "BETA_barra", "HSIGMA_barra", "HALPHA_barra",
            "DASTD_barra", "CMRA_barra",
            "STOM_barra", "STOQ_barra", "STOA_barra",
            "RSTR_barra"
            ]

    # Barra财务类因子原始指标和目标指标
    barra_finance_indicators = [
            "mkt_cap_ard", "longdebttodebt", "other_equity_instruments_PRE",
            "tot_equity", "tot_liab", "tot_assets", "pb_lf",
            "pe_ttm", "pcf_ocf_ttm", "eps_ttm", "orps"
            ]
    barra_finance_target_indicators = [
            "MLEV_barra", "BLEV_barra", "DTOA_barra", "BTOP_barra",
            "ETOP_barra", "CETOP_barra", "EGRO_barra", "SGRO_barra"
            ]

    # 技术指标参数设置
    _tech_params = {
                    "BIAS": [20],
                    "MACD": [10, 30, 15],
                    "PSY": [20],
                    "RSI": [20],
                    }
    freqmap = {}

    def __init__(self):
        """初始化函数
        调用__update_frepmap方法更新频率映射字典
        """
        self.__update_frepmap()

    # 更新频率映射字典方法
    def __update_frepmap(self):
        """更新频率映射字典方法
        遍历root目录下的所有文件，将文件名(不含扩展名)作为key，root路径作为value
        更新到freqmap字典中
        """
        self.freqmap.update({name.split(".")[0]: self.root for name in os.listdir(self.root)})

    # 打开文件方法
    def open_file(self, name):
        '''打开文件方法
        根据传入的文件名name，从freqmap字典中获取对应的路径path
        如果path为None，则抛出异常
        否则，使用pandas的read_csv方法读取文件，并返回数据
        Args:
            name: 文件名
        Returns:
            数据
        Raises:
            Exception: 如果path为None，则抛出异常
        '''
        if name == 'meta':
            return pd.read_excel(os.path.join(self.root, 'src', self.metafile), index_col=[0], parse_dates=['ipo_date', "delist_date"])
        elif name == 'month_map':
            return pd.read_excel(os.path.join(self.root, 'src', self.mmapfile), index_col=[0], parse_dates=[0, 1])['calendar_date']
        elif name == 'trade_days_begin_end_of_month':
            return pd.read_excel(os.path.join(self.root, 'src', self.tdays_be_m_file), index_col=[1], parse_dates=[0, 1])
        elif name == 'month_group':
            return pd.read_excel(os.path.join(self.root, 'src', self.month_group_file), index_col=[0], parse_dates=True)
        elif name == 'tradedays':
            return pd.read_excel(os.path.join(self.root, 'src', self.tradedays_file), index_col=[0], parse_dates=True).index.tolist()
        path = self.freqmap.get(name, None) # 从字典中获取路径
        if path is None:
            raise Exception(f'{name} 无法识别或不在文件目录中，请检查后重试。')
        try:
            dat = pd.read_csv(os.path.join(path, name+'.csv'), index_col=[0], engine='python', encoding='gbk') 
            dat = pd.DataFrame(data=dat, index=dat.index.union(self.meta.index), columns=dat.columns)
        except TypeError:
            print(name, path)
            raise
        dat.columns = pd.to_datetime(dat.columns)
        #if name in ('stm_issuingdate', 'applied_rpt_date_M'):
        #    dat = dat.replace('0', np.nan)
        #    dat = dat.applymap(pd.to_datetime)
        return dat

    # 关闭文件方法
    def close_file(self, df, name, **kwargs):
        """关闭文件方法
        根据传入的文件名name，从freqmap字典中获取对应的路径path
        如果path为None，则抛出异常
        否则，使用pandas的to_csv方法将数据保存为csv文件
        Args:
            df: 数据
            name: 文件名
        Raises:
            Exception: 如果path为None，则抛出异常
        """
        if name == 'meta':
            # 构造文件路径
            file_path = os.path.join(self.root, 'src', self.metafile)
            # 检查并创建目标目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 写入Excel文件
            df.to_excel(file_path, encoding='gbk', **kwargs)
        elif name == 'month_map':
            file_path = os.path.join(self.root, 'src', self.mmapfile)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_excel(file_path, encoding='gbk', **kwargs)
        elif name == 'trade_days_begin_end_of_month':
            file_path = os.path.join(self.root, 'src', self.tdays_be_m_file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_excel(file_path, encoding='gbk', **kwargs)
        elif name == 'tradedays':
            file_path = os.path.join(self.root, 'src', self.tradedays_file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_excel(file_path, encoding='gbk', **kwargs)
        else:
            path = self.freqmap.get(name, None)
            if path is None:
                path = self.root

            #if name in ['stm_issuingdate', 'applied_rpt_date_M']:
            #    df = df.replace(0, pd.NaT)
            file_path = os.path.join(path, name + '.csv')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, encoding='gbk', **kwargs)
            self.__update_frepmap()
        self.__update_attr(name)

    # 处理序列中的缺失值(NaN)
    @staticmethod
    def _fill_nan(series, value=0, ffill=False):
        """处理序列中的缺失值(NaN)
        Args:
            series (pd.Series): 需要处理的pandas序列
            value (int/float): 填充值，默认为0
            ffill (bool): 是否使用前向填充，默认为False
        Returns:
            pd.Series: 处理后的序列
            
        功能说明:
            1. 当ffill=True时，使用前向填充方法处理缺失值
            2. 当ffill=False且value!=0时，从第一个有效值开始向后填充指定值
            3. 当ffill=False且value=0时，不做任何处理
        """
        if ffill:  # 前向填充模式
            series = series.fillna(method='ffill')  # 使用前向填充处理缺失值
        else:  # 非前向填充模式
            if value:  # 如果指定了填充值
                # 找到第一个非空值的索引位置
                start_valid_idx = np.where(pd.notna(series))[0][0]  
                # 从第一个有效值开始向后填充指定值
                series.loc[start_valid_idx:] = series.loc[start_valid_idx:].fillna(0)  
        return series  # 返回处理后的序列

    # 更新实例属性方法
    def __update_attr(self, name):
        """更新实例属性方法
        用于动态更新实例属性，确保属性值与文件数据同步
        
        Args:
            name (str): 需要更新的属性名
        """
        if name in self.__dict__:  # 如果属性已存在于实例字典中
            del self.__dict__[name]  # 先删除旧属性
        self.__dict__[name] = getattr(self, name, None)  # 重新获取属性值并更新

    # Python特殊方法 - 属性访问拦截器
    def __getattr__(self, name):
        """Python特殊方法 - 属性访问拦截器
        当访问实例不存在的属性时自动调用此方法
        Args:
            name (str): 尝试访问的属性名       
        Returns:
            从文件加载的属性值
            
        实现逻辑:
            1. 检查属性是否已存在于实例字典中
            2. 如果不存在，则调用open_file方法加载对应文件数据
            3. 将加载的数据存入实例字典并返回
        """
        if name not in self.__dict__:  # 检查属性是否已缓存
            self.__dict__[name] = self.open_file(name)  # 动态加载文件数据并缓存
        return self.__dict__[name]  # 返回属性值

class FactorGenerater:
    def __init__(self, using_fetch=False):
        self.data = Data()
        if not using_fetch:
            self.dates_d = sorted(self.adjfactor.columns)
            self.dates_m = sorted(self.pct_chg_M.columns)

    def __getattr__(self, name):
        # 如果属性不存在，则尝试从 data 对象中获取
        return getattr(self.data, name, None)

    # 获取交易日列表
    def _get_trade_days(self, startday, endday, freq=None):
        """获取交易日列表
        Args:
            startday: 开始日期
            endday: 结束日期
            freq: 频率，默认为None
        Returns:
            交易日列表

        功能说明:
            1. 根据传入的开始日期、结束日期和频率，返回对应的交易日列表
            2. 如果频率为None，则使用实例的频率属性
            3. 如果频率为'd'，则直接返回交易日列表
            4. 如果频率为其他值，则根据交易日列表和月份映射表，生成新的交易日列表
        """
        if freq is None:
            freq = self.freq  # 使用实例的频率属性
        datelist = sorted(self.tradedays)  # 获取交易日列表
        startday, endday = pd.to_datetime((startday, endday)) 
        if freq == 'd':  # 检查频率是否为日频('d')
            try: 
                # 尝试获取开始日期在交易日列表中的索引位置
                start_idx = self._get_date_idx(startday, datelist)
            except IndexError:
                # 如果开始日期不在交易日列表中，返回空列表
                return []
            else:
                try:
                    # 尝试获取结束日期在交易日列表中的索引位置
                    end_idx = self._get_date_idx(endday, datelist)
                except IndexError:
                    # 如果结束日期不在交易日列表中，返回从开始日期到最后一个交易日的列表
                    return datelist[start_idx:]
                else:
                    # 如果开始和结束日期都在交易日列表中，返回两者之间的交易日列表(包含结束日期)
                    return datelist[start_idx:end_idx+1]
        else:  # 非日频(d)的其他频率处理逻辑
            # 1. 对交易日列表按指定频率重新采样
            new_cdays_curfreq = pd.Series(index=datelist).resample(freq).asfreq().index
            
            # 2. 创建日历日到交易日的映射字典
            c_to_t_dict = {cday:tday for tday, cday in self.month_map.to_dict().items()}
            
            try:
                # 3. 将重新采样后的日历日转换为交易日
                new_tdays_curfreq = [c_to_t_dict[cday] for cday in new_cdays_curfreq]
            except KeyError:
                # 4. 如果转换失败(最后一个日历日无对应交易日)，则排除最后一个日历日
                new_tdays_curfreq = [c_to_t_dict[cday] for cday in new_cdays_curfreq[:-1]]
            
            # 5. 获取开始日期在转换后交易日列表中的索引(加1跳过第一个)
            start_idx = self._get_date_idx(c_to_t_dict.get(startday, startday), new_tdays_curfreq) + 1
            
            try:
                # 6. 获取结束日期在转换后交易日列表中的索引
                end_idx = self._get_date_idx(c_to_t_dict.get(endday, endday), new_tdays_curfreq)
            except IndexError:
                # 7. 如果结束日期超出范围，则使用最后一个交易日
                end_idx = len(new_tdays_curfreq) - 1
            
            # 8. 返回指定范围内的交易日列表(包含结束日期)
            return new_tdays_curfreq[start_idx:end_idx+1]

    @lazyproperty
    def trade_days(self):
        """交易日列表属性(延迟加载)
        使用lazyproperty装饰器实现延迟加载模式，首次访问时才会计算并缓存结果
        
        实现逻辑:
            1. 调用_get_trade_days方法获取指定日期范围内的交易日列表
            2. 将结果缓存在__trade_days私有属性中
            3. 返回缓存的交易日列表
        注意:
            - 使用装饰器确保只计算一次
            - 计算结果会被缓存以提高后续访问效率
        """
        self.__trade_days = self._get_trade_days(self.startday, self.endday)
        return self.__trade_days

    # 保存文件
    def save_file(self, datdf, path):
        datdf = datdf.loc[~pd.isnull(datdf['is_open1']), :] # 剔除停牌的股票

        for col in ['name', 'industry_sw']:
            datdf[col] = datdf[col].apply(str)
        datdf = datdf.loc[~datdf['name'].str.contains('0')]

        save_cond1 = (~datdf['name'].str.contains('ST')) #剔除ST股票
        save_cond2 = (~pd.isnull(datdf['industry_sw'])) & (~datdf['industry_sw'].str.contains('0')) #剔除行业值为0或为空的股票
        save_cond3 = (~pd.isnull(datdf['MKT_CAP_FLOAT'])) #剔除市值为空的股票
        save_cond = save_cond1 & save_cond2 & save_cond3
        datdf = datdf.loc[save_cond]

        datdf = datdf.reset_index()
        datdf.index = range(1, len(datdf)+1)
        datdf.index.name = 'No'
        datdf = datdf.rename(columns={"index":"code"})

        #之前不管是计算指标还是计算因子,当某些除法操作分母为0的情况会导致产生inf值,所以这里统一处理
        datdf = datdf.replace(np.inf, 0).replace(-np.inf, 0) # 将inf替换为0
        # 将所有列的数据类型转换为float64
        if path.endswith('.csv'):
            return datdf.to_csv(path, encoding='gbk')
        else:
            raise TypeError("不支持的文件类型 {}，目前仅支持csv格式".format(path.split('.')[-1]))

    # 合并两个DataFrame对象
    @staticmethod
    def concat_df(left, right, *, how="outer", left_index=True, right_index=True, **kwargs):
        """合并两个DataFrame对象
        Args:
            left (pd.DataFrame): 左侧DataFrame对象
            right (pd.DataFrame): 右侧DataFrame对象
            how (str): 合并方式，默认为"outer"
            left_index (bool): 左侧DataFrame的索引是否作为合并键，默认为True
            right_index (bool): 右侧DataFrame的索引是否作为合并键，默认为True
            **kwargs: 其他参数
        Returns:
            pd.DataFrame: 合并后的DataFrame对象
        """
        return pd.merge(left, right, how=how, left_index=left_index, right_index=right_index, **kwargs)

    # 创建因子文件
    def create_factor_file(self, date, savepath):
        """创建因子数据文件方法
        Args:
            date (str): 日期字符串，格式为YYYYMMDD
            savepath (str): 文件保存路径
        Raises:
            FileAlreadyExistError: 如果目标文件已存在则抛出此异常
            
        功能说明:
            1. 检查目标文件是否已存在，存在则抛出异常
            2. 获取基础数据(stklist, dat0)和因子数据(dat1)
            3. 合并两个数据集
            4. 将合并结果保存到指定路径
        """
        if os.path.exists(savepath):
            raise FileAlreadyExistError(f"{date}的数据已存在，请尝试调用update方法。")
        stklist, dat0 = self.get_basic_data(date)  # 获取基础数据
        dat1 = self.get_factor_data(date, stklist)  # 获取因子数据
        res = self.concat_df(dat0, dat1)  # 合并数据集
        self.save_file(res, savepath)  # 保存结果文件

    # 获取基础数据
    def get_basic_data(self, tdate):
        """获取基础数据方法
        Args:
            tdate (str): 日期字符串，格式为YYYYMMDD
        Returns:
            stklist (list): 股票列表
            dat0 (pd.DataFrame): 基础数据
        
        功能说明:
            1. 获取指定日期的股票列表
            2. 根据股票列表获取基础数据
            3. 处理基础数据中的缺失值
            4. 返回股票列表和基础数据
        """
        tdate = pd.to_datetime(tdate)
        stklist = self.meta.index.tolist() # 股票列表
        df0 = self.meta[self.meta['ipo_date'] <= tdate] #股票上市时间早于指定时间
        cond = (pd.isnull(df0['delist_date'])) | (df0['delist_date'] >= tdate) #股票退市时间晚于指定时间
        df0 = df0[cond] 
        #接下来还需要判断如果每月停牌日期大于一定数目就排除这只股票
        bdate = self.trade_days_begin_end_of_month.at[tdate, 'month_start'] # 当月月初
        tradestatus = self.trade_status.loc[df0.index, bdate:tdate] 
        tradestatus = (tradestatus==0) #停牌的股票为True
        cond = (tradestatus.sum(axis=1) < 10) #停牌日期小于10天的股票才入选, 超过10天的排除
        df0 = df0[cond]
        df0 = df0.rename(columns={'sec_name':'name'}) 
        del df0['delist_date'] # 删除退市日期列
        stocklist = df0.index.tolist() # 股票列表
        caldate = self.month_map[tdate] # 交易日
        df0["industry_zx"] = self.industry_citic.loc[stocklist, caldate] #中信行业分类
        df0["industry_sw"] = self.industry_sw.loc[stocklist, caldate] #申万行业分类
        df0['MKT_CAP_FLOAT'] = self.mkt_cap_float_m.loc[stocklist, caldate] # 流通市值
        try:
            tdate = self._get_next_month_first_trade_date(tdate) #下个月第一个交易日
        except IndexError:
            df0["is_open1"] = None
            df0["PCT_CHG_NM"] = None
            return stocklist, df0
        df0["is_open1"] = self.trade_status.loc[stocklist, tdate].map({1:"TRUE", 0:"FALSE"}) #下个月是否开盘
        df0["PCT_CHG_NM"] = self.get_next_pctchg(stocklist, tdate) #下月的月收益率,回测的时候会使用到
        return stocklist, df0

    # 获取下个月第一个交易日的日期
    def _get_next_month_first_trade_date(self, date):
        """获取下个月第一个交易日的日期
        Args:
            date (str): 日期字符串，格式为YYYYMMDD
        Returns:
            str: 下个月第一个交易日的日期字符串，格式为YYYYMMDD

        功能说明:
            1. 将输入的日期字符串转换为datetime对象
            2. 获取交易日列表
            3. 找到交易日列表中第一个大于输入日期的交易日
            4. 返回该交易日的日期字符串
        """
        date = pd.to_datetime(date)
        tdates = self.trade_status.columns.tolist() # 交易日列表
        # 找到交易日列表中第一个大于输入日期的交易日
        def _if_same_month(x):
            nonlocal date
            # 如果不是12月，就判断是不是下个月，如果是12月，就判断是不是明年的1月
            
            if date.month != 12: 
                return (x.year != date.year) or (x.month - 1 != date.month)
            else:
                return (x.year - 1 != date.year) or (x.month != 1)
        # 使用dropwhile函数过滤掉不是下个月的交易日
        daterange = dropwhile(_if_same_month, tdates)
        return list(daterange)[0]

    # 获取下个月的月收益率
    def get_next_pctchg(self, stocklist, tdate):
        """获取下个月的月收益率
        Args:
            stocklist (list): 股票列表
            tdate (str): 日期字符串，格式为YYYYMMDD
        Returns:
            list: 下个月的月收益率列表

        功能说明:
            1. 将输入的日期字符串转换为datetime对象
            2. 计算下个月的日期
            3. 获取下个月的月收益率
            4. 返回该月收益率列表
        """
        try:
            nextdate = tdate + toffsets.MonthEnd(1)
            dat = self.pct_chg_M.loc[stocklist, nextdate] # 下个月的月收益率
        except Exception as e:
            print("获取下个月数据失败。错误信息: {}".format(e))
            dat = [np.nan] * len(stocklist)
        return dat

    # 获取上一个月的最后一个交易日
    def get_last_month_end(self, date):
        """获取上一个月的最后一个交易日
        Args:
            date (str): 日期字符串，格式为YYYYMMDD
        Returns:
            str: 上一个月的最后一个交易日的日期字符串，格式为YYYYMMDD

        功能说明:
            1. 将输入的日期字符串转换为datetime对象
            2. 计算上一个月的最后一个交易日
            3. 返回该交易日的日期字符串
        """
        if date.month == 1:
            lstyear = date.year - 1
            lstmonth = 12
        else:
            lstyear = date.year
            lstmonth = date.month - 1
        return datetime(lstyear, lstmonth, 1) + toffsets.MonthEnd(n=1)

    # 获取因子数据
    def get_factor_data(self, tdate, stocklist):
        """获取因子数据方法
        Args:
            tdate (str): 日期字符串，格式为YYYYMMDD
            stocklist (list): 股票列表
        Returns:
            pd.DataFrame: 因子数据

        功能说明:
            1. 获取指定日期的因子数据
            2. 处理因子数据中的缺失值
            3. 返回因子数据
        """
        caldate = self.month_map[tdate] # 交易日
        dat1 = self._get_value_data(stocklist, caldate) # 价值数据
        dat2 = self._get_growth_data(stocklist, caldate) # 成长数据
        dat3 = self._get_finance_data(stocklist, caldate) # 财务数据
        dat4 = self._get_leverage_data(stocklist, caldate) # 杠杆数据
        dat5 = self._get_cal_data(stocklist, tdate) # 日历数据
        dat6 = self._get_tech_data(stocklist, tdate) # 技术数据
        res = reduce(self.concat_df, [dat1, dat2, dat3, dat4, dat5, dat6]) # 合并数据
        dat7 = self._get_barra_quote_data(stocklist, tdate) # 因子数据
        dat8 = self._get_barra_finance_data(stocklist, tdate) # 因子数据
        res = reduce(self.concat_df, [res, dat7, dat8]) 
        return res

    # 获取价值数据(self.value_target_indicators)
    def _get_value_data(self, stocks, caldate):
        """
            Default value indicators getted from windpy:
            'pe_ttm', 'val_pe_deducted_ttm', 'pb_lf', 'ps_ttm', 
            'pcf_ncf_ttm', 'pcf_ocf_ttm', 'dividendyield2', 'profit_ttm'
            
            Default target value indicators:
            'EP', 'EPcut', 'BP', 'SP', 
            'NCFP', 'OCFP', 'DP', 'G/PE'
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat['EP'] = 1 / self.pe_ttm_m.loc[stocks, date] # 每股收益
        dat['EPcut'] = 1 / self.val_pe_deducted_ttm_m.loc[stocks, date] # 每股收益(扣除非经常性损益)
        dat['BP'] = 1 / self.pb_lf_m.loc[stocks, date] # 市净率
        dat['SP'] = 1 / self.ps_ttm_m.loc[stocks, date] # 市销率 
        dat['NCFP'] = 1 / self.pcf_ncf_ttm_m.loc[stocks, date] # 每股经营现金流
        dat['OCFP'] = 1 / self.pcf_ocf_ttm_m.loc[stocks, date] # 每股经营现金流(扣除非经常性损益)
        dat['DP'] = self.dividendyield2_m.loc[stocks, date] # 股息率
        dat['G/PE'] = self.profit_ttm_G_m.loc[stocks, date] * dat['EP'] # 收益增长率

        dat = dat[self.value_target_indicators] # 目标价值指标
        return dat

    # 获取成长数据(self.growth_target_indicators)
    def _get_growth_data(self, stocks, caldate):
        """
            Default growth indicators getted from windpy:
            "qfa_yoysales", "qfa_yoyprofit", "qfa_yoyocf", "qfa_roe"
            
            Default target growth indicators:
            "Sales_G_q","Profit_G_q", "OCF_G_q", "ROE_G_q", 
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["Sales_G_q"] = self.qfa_yoysales_m.loc[stocks, date] # 营业收入增长率
        dat["Profit_G_q"] = self.qfa_yoyprofit_m.loc[stocks, date] # 净利润增长率
        dat["OCF_G_q"] = self.qfa_yoyocf_m.loc[stocks, date] # 经营现金流增长率
        dat['ROE_G_q'] = self.qfa_roe_G_m.loc[stocks, date] # 净资产收益率增长率

        dat = dat[self.growth_target_indicators] # 目标成长指标
        return dat

    # 获取财务数据
    def _get_finance_data(self, stocks, caldate):
        """
            Default finance indicators getted from windpy:
            "roe_ttm2_m", "qfa_roe_m", 
            "roa2_ttm2_m", "qfa_roa_m", 
            "grossprofitmargin_ttm2_m", "qfa_grossprofitmargin_m", 
            "deductedprofit_ttm", "qfa_deductedprofit_m", "or_ttm", "qfa_oper_rev_m", 
            "turnover_ttm_m", "qfa_netprofitmargin_m", 
            "ocfps_ttm", "eps_ttm", "qfa_net_profit_is_m", "qfa_net_cash_flows_oper_act_m"
            
            Default target finance indicators:
            "ROE_q", "ROE_ttm", 
            "ROA_q", "ROA_ttm", 
            "grossprofitmargin_q", "grossprofitmargin_ttm", 
            "profitmargin_q", "profitmargin_ttm",
            "assetturnover_q", "assetturnover_ttm", 
            "operationcashflowratio_q", "operationcashflowratio_ttm"
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["ROE_q"] = self.qfa_roe_m.loc[stocks, date] # 净资产收益率
        dat["ROE_ttm"] = self.roe_ttm2_m.loc[stocks, date] # 净资产收益率(同比增长率)

        dat["ROA_q"] = self.qfa_roa_m.loc[stocks, date] # 总资产收益率
        dat["ROA_ttm"] = self.roa2_ttm2_m.loc[stocks, date] # 总资产收益率(同比增长率)

        dat["grossprofitmargin_q"] = self.qfa_grossprofitmargin_m.loc[stocks, date] # 毛利率
        dat["grossprofitmargin_ttm"] = self.grossprofitmargin_ttm2_m.loc[stocks, date] # 毛利率(同比增长率)

        #dat["profitmargin_q"] = self.qfa_deductedprofit_m.loc[stocks, date] / self.qfa_oper_rev_m.loc[stocks, date]
        #dat["profitmargin_ttm"] = self.deductedprofit_ttm.loc[stocks, date] / self.or_ttm.loc[stocks, date]

        #dat["assetturnover_q"] = self.qfa_roa_m.loc[stocks, date] / self.qfa_netprofitmargin_m.loc[stocks, date]
        dat['assetturnover_ttm'] = self.turnover_ttm_m.loc[stocks, date] # 资产周转率

        #dat["operationcashflowratio_q"] = self.qfa_net_cash_flows_oper_act_m.loc[stocks, date] / self.qfa_net_profit_is_m.loc[stocks, date]
        #dat["operationcashflowratio_ttm"] = self.ocfps_ttm.loc[stocks, date] / self.eps_ttm.loc[stocks, date]

        #dat = dat[self.finance_target_indicators]
        return dat

    # 获取杠杆数据(self.leverage_target_indicators)
    def _get_leverage_data(self, stocks, caldate):
        """
            Default leverage indicators getted from windpy:
            "assetstoequity_m", "longdebttoequity_m", "cashtocurrentdebt_m", "current_m"
            
            Default target leverage indicators:
            "financial_leverage", "debtequityratio", "cashratio", "currentratio"
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["financial_leverage"] = self.assetstoequity_m.loc[stocks, date] # 资产负债率
        dat["debtequityratio"] = self.longdebttoequity_m.loc[stocks, date] # 负债率
        dat["cashratio"] = self.cashtocurrentdebt_m.loc[stocks, date] # 现金比率
        dat["currentratio"] = self.current_m.loc[stocks, date] # 流动比率

        dat = dat[self.leverage_target_indicators] # 目标杠杆指标

        return dat

    # 获取日历数据
    def _get_cal_data(self, stocks, tdate):
        """
            Default calculated indicators getted from windpy:
            "mkt_cap_float", "holder_avgpct", "holder_num"
            
            Default target calculated indicators:
            "ln_capital", 
            "HAlpha", 
            "return_1m", "return_3m", "return_6m", "return_12m", 
            "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
            "exp_wgt_return_1m",  "exp_wgt_return_3m",  "exp_wgt_return_6m", "exp_wgt_return_12m", 
            "std_1m", "std_3m", "std_6m", "std_12m", 
            "beta", 
            "turn_1m", "turn_3m", "turn_6m", "turn_12m", 
            "bias_turn_1m", "bias_turn_3m", "bias_turn_6m", "bias_turn_12m", 
            "holder_avgpctchange"
        """
        tdate = pd.to_datetime(tdate)
        dat = pd.DataFrame(index=stocks)

        caldate = self.month_map[tdate]

        dat['ln_capital'] = np.log(self.mkt_cap_float_m.loc[stocks, caldate])
        #dat['holder_avgpctchange'] = self.holder_avgpctchg.loc[stocks, caldate]

        dat1 = self._get_mom_vol_data(stocks, tdate, self.dates_d, params=[1,3,6,12])
        dat2 = self._get_turnover_data(stocks, tdate, self.dates_d, params=[1,3,6,12])
        dat3 = self._get_regress_data(stocks, tdate, self.dates_m, params=["000001.SH", 24])

        dat = reduce(self.concat_df, [dat, dat1, dat2, dat3])
        #dat = dat[self.cal_target_indicators]
        return dat

    # 获取技术数据
    def _get_tech_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "close(freq=d)"
            
            Default target technique indicators:
            "MACD", "DEA", "DIF", "RSI", "PSY", "BIAS"
        """
        dat = pd.DataFrame(index=stocks)
        for tname in self.tech_indicators:
            calfunc = getattr(self, 'cal_'+tname, None)
            if calfunc is None:
                msg = "Please define property:'{}' first.".format("cal_"+tname)
                raise NotImplementedError(msg)
            else:
                if tname == "MACD":
                    dat["DIF"], dat["DEA"], dat["MACD"] = calfunc(stocks, tdate, self._tech_params[tname])
                else:
                    dat[tname] = calfunc(stocks, tdate, self._tech_params[tname])
        return dat

    # 获取动量和波动率数据
    def _get_mom_vol_data(self, stocks, tdate, dates, params=(1,3,6,12)):
        """获取动量和波动率相关数据
        Args:
            stocks: 股票代码列表
            tdate: 交易日期
            dates: 日期列表
            params: 时间窗口参数,默认为(1,3,6,12)个月
        Returns:
            pd.DataFrame: 包含以下指标的DataFrame:
                - return_Nm: N个月的收益率
                - wgt_return_Nm: N个月的加权收益率(用换手率加权)
                - exp_wgt_return_Nm: N个月的指数加权收益率
                - std_Nm: N个月的收益率标准差
        """
        pct_chg = self.pct_chg # 收益率
        turnover = self.turn # 换手率
        caldate = self.month_map[tdate]
        res = pd.DataFrame(index=stocks)
        # 对每个时间窗口计算相关指标
        for offset in params:
            # 获取时间窗口内的日期列表
            period_d = self._get_period_d(tdate, offset=-offset, freq="M", datelist=dates)
            cur_pct_chg_d = pct_chg.loc[stocks, period_d] # 收益率
            cur_turnover = turnover.loc[stocks, period_d] # 换手率
            wgt_pct_chg = cur_pct_chg_d * cur_turnover # 计算换手率加权的收益率
            # 计算时间衰减权重
            days_wgt = cur_pct_chg_d.expanding(axis=1).apply(lambda df: np.exp(-(len(period_d) - len(df))/4/offset))
            exp_wgt_pct_chg = wgt_pct_chg * days_wgt # 计算时间加权的收益率
            cur_pct_chg_m = getattr(self, f"pctchg_{offset}M", None) # 获取月度收益率数据
            res[f"return_{offset}m"] = cur_pct_chg_m.loc[stocks, caldate]  # 原始收益率
            res[f"wgt_return_{offset}m"] = wgt_pct_chg.apply(np.nanmean, axis=1)  # 换手率加权收益率
            res[f"exp_wgt_return_{offset}m"] = exp_wgt_pct_chg.apply(np.nanmean, axis=1)  # 时间加权收益率
            res[f"std_{offset}m"] = cur_pct_chg_d.apply(np.nanstd, axis=1)  # 收益率标准差
            
        return res

    # 获取换手率数据
    def _get_turnover_data(self, stocks, tdate, dates, params=(1,3,6,12)):
        base_period_d = self._get_period_d(tdate, offset=-2, freq="y", datelist=dates)
        cur_turnover_base = self.turn.loc[stocks, base_period_d]
        turnover_davg_base = cur_turnover_base.apply(np.nanmean, axis=1)
        res = pd.DataFrame(index=stocks)
        for offset in params:
            period_d = self._get_period_d(tdate, offset=-offset, freq="M", datelist=dates)
            cur_turnover = self.turn.loc[stocks, period_d]
            turnover_davg = cur_turnover.apply(np.nanmean, axis=1)
            res[f"turn_{offset}m"] = turnover_davg
            res[f"bias_turn_{offset}m"] = turnover_davg / turnover_davg_base - 1
        return res

    # 获取回归数据
    def _get_regress_data(self, stocks, tdate, dates, params=("000001.SH", 60)):
        """
            return value contains:
            HAlpha --intercept
            beta   --slope
        """
        index_code, period = params

        col_index = self._get_period(tdate, offset=-period, freq="M", datelist=dates, resample=False) #前推60个月(五年)
        pct_chg_idx = self.pct_chg_M.loc[index_code, col_index]
        pct_chg_m = self.pct_chg_M.loc[stocks, col_index].dropna(how='any', axis=0).T
        x, y = pct_chg_idx.values.reshape(-1,1), pct_chg_m.values

        valid_stocks = pct_chg_m.columns.tolist()
        try:
            beta, Halpha = self.regress(x, y)
        except ValueError as e:
            print(e)
            #raise
            beta, Halpha = np.empty((len(valid_stocks),1)), np.empty((1, len(valid_stocks)))

        beta = pd.DataFrame(beta, index=valid_stocks, columns=['beta'])
        Halpha = pd.DataFrame(Halpha.T, index=valid_stocks, columns=['HAlpha'])
        res = self.concat_df(beta, Halpha)
        return res

    # 获取barra_quote数据(self.barra_quote_target_indicators)
    def _get_barra_quote_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "mkt_cap_float", "pct_chg", "amt"
            
            Default target barra_quote indicators:
            "LNCAP_barra", "MIDCAP_barra", 
            "BETA_barra", "HSIGMA_barra", "HALPHA_barra",
            "DASTD_barra", "CMRA_barra",
            "STOM_barra", "STOQ_barra", "STOA_barra",
            "RSTR_barra"
        """
        tdate = pd.to_datetime(tdate)
        caldate = self.month_map[tdate]
        dat = pd.DataFrame(index=stocks)

        dat1 = self._get_size_barra(stocks, caldate, self.dates_d, params=[True,True,True])
        dat2 = self._get_regress_barra(stocks, tdate, self.dates_d, params=[4,504,252,True,'000300.SH'])
        dat3 = self._get_dastd_barra(stocks, tdate, self.dates_d, params=[252,42])
        dat4 = self._get_cmra_barra(stocks, tdate, self.dates_d, params=[12, 21])
        dat5 = self._get_liquidity_barra(stocks, tdate, params=[21,1,3,12])
        dat6 = self._get_rstr_barra(stocks, tdate, self.dates_d, params=[252,126,11,'000300.SH'])

        dat = reduce(self.concat_df, [dat, dat1, dat2, dat3, dat4, dat5, dat6])
        dat = dat[self.barra_quote_target_indicators]
        return dat

    # 获取barra_quote数据中的市值数据
    def _get_size_barra(self, stocks, caldate, dates, params=(True,True,True)):
        intercept, standardize, wls = params

        res = pd.DataFrame(index=stocks)
        lncap = self.mkt_cap_float_m.loc[stocks, caldate].apply(np.log)
        lncap_3 = lncap ** 3

        if wls:
            w = lncap.apply(np.sqrt)
            x_y_w = pd.concat([lncap, lncap_3, w], axis=1).dropna(how='any', axis=0)
            x, y, w = x_y_w.iloc[:,0], x_y_w.iloc[:,1], x_y_w.iloc[:,-1]
            x, y, w = x.values, y.values, w.values
        else:
            w = 1
            x_and_y = pd.concat([lncap, lncap_3], axis=1).dropna(how='any', axis=0)
            x, y = x_and_y.iloc[:,0], x_and_y.iloc[:,-1]
            x, y = x.values, y.values

        intercept, coef = self.regress(x, y, intercept, w)
        resid = lncap_3 - (coef * lncap + intercept)

        if standardize:
            resid = self.standardize(self.winsorize(resid))
        res['MIDCAP_barra'] = resid
        res['LNCAP_barra'] = lncap
        return res

    # 获取barra_quote数据中的回归数据
    def _get_regress_barra(self, stocks, tdate, dates_d, params=(4,504,252,True,'000300.SH')):
        """计算Barra回归相关因子
        Args:
            stocks: 股票代码列表
            tdate: 交易日期
            dates_d: 日期序列
            params: 参数元组,包含:
                shift: 移动窗口数(默认4)
                window: 回归窗口长度(默认504)
                half_life: 半衰期(默认252)
                if_intercept: 是否包含截距项(默认True)
                index_code: 基准指数代码(默认'000300.SH')
        Returns:
            包含BETA、HALPHA和HSIGMA三个Barra因子的DataFrame
        """
        shift, window, half_life, if_intercept, index_code = params # 解析参数
        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life) # 计算指数加权
        idx = self._get_date_idx(tdate, dates_d) # 获取日期索引
        date_period = dates_d[idx-window+1-shift:idx+1] # 获取日期区间
        pct_chgs = self.pct_chg.T.loc[date_period,:] # 获取收益率数据
        # 循环计算每个移动窗口的回归系数
        for i in range(1,shift+1):
            # 获取当前窗口的收益率数据
            pct_chg = pct_chgs.iloc[i:i+window,:]
            x = pct_chg.loc[:, index_code]  # 基准指数收益率
            ys = pct_chg.loc[:, stocks].dropna(how='any', axis=1)  # 个股收益率
            X, Ys = x.values, ys.values
            # 进行回归计算
            try:
                intercept, coef = self.regress(X, Ys, if_intercept, w)
            except:
                print(X)
                print(Ys)
                raise

            alpha = pd.Series(intercept, index=ys.columns)
            beta = pd.Series(coef[0], index=ys.columns)
            alpha.name = f'alpha_{i}'; beta.name = f'beta_{i}'
            res = pd.concat([res, alpha, beta], axis=1)
            # 在最后一个窗口计算残差标准差
            if i == shift:
                resid = Ys - (intercept + X.reshape(-1,1) @ coef)
                sigma = pd.Series(np.std(resid, axis=0), index=ys.columns)
                sigma.name = 'HSIGMA_barra'
                res = pd.concat([res, sigma], axis=1)
        # 计算最终的Barra因子
        res['HALPHA_barra'] = np.sum((res[f'alpha_{i}'] for i in range(1,shift+1)), axis=0)  # Alpha因子
        res['BETA_barra'] = np.sum((res[f'beta_{i}'] for i in range(1,shift+1)), axis=0)  # Beta因子
        # 只保留三个Barra因子
        res = res[['BETA_barra', 'HALPHA_barra', 'HSIGMA_barra']]
        return res

    # 获取barra_quote数据中的波动率数据
    def _get_dastd_barra(self, stocks, tdate, dates_d, params=(252,42)):
        window, half_life = params

        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life)
        pct_chg = self._get_daily_data("pct_chg", stocks, tdate, window, dates_d)
        pct_chg = pct_chg.dropna(how='any', axis=1)
        res['DASTD_barra'] = pct_chg.apply(self._std_dev, args=(w,))
        return res

    @staticmethod
    def _std_dev(series, weight=1):
        mean = np.mean(series)
        std_dev = np.sqrt(np.sum((series - mean)**2 * weight))
        return std_dev

    def _get_cmra_barra(self, stocks, tdate, dates_d, params=(12,21)):
        months, days_pm = params
        window = months * days_pm

        res = pd.DataFrame(index=stocks)
        pct_chg = self._get_daily_data("pct_chg", stocks, tdate, window, dates_d)
        pct_chg = pct_chg.dropna(how='any', axis=1)
        res['CMRA_barra'] = np.log(1 + pct_chg).apply(self._cal_cmra, args=(months, days_pm))
        return res

    # 计算cmra
    @staticmethod
    def _cal_cmra(series, months=12, days_per_month=21):
        z = sorted(series[-i * days_per_month:].sum() for i in range(1, months+1))
        return z[-1] - z[0]

    # 获取barra_quote数据中的流动性数据
    def _get_liquidity_barra(self, stocks, tdate, params=(21,1,3,12)):
        days_pm, freq1, freq2, freq3 = params
        window = freq3 * days_pm

        res = pd.DataFrame(index=stocks)
        amt = self._get_daily_data('amt', stocks, tdate, window)
        mkt_cap_float = self._get_daily_data('mkt_cap_float', stocks, tdate, window)
        share_turnover = amt / mkt_cap_float

        for freq in [freq1, freq2, freq3]:
            res[f'st_{freq}'] = share_turnover.iloc[-freq*days_pm:,:].apply(self._cal_liquidity, args=(freq,))
        res = res.rename(columns={f'st_{freq1}':'STOM_barra',
                                  f'st_{freq2}':'STOQ_barra',
                                  f'st_{freq3}':'STOA_barra'})
        return res

    @staticmethod
    def _cal_liquidity(series, freq=1):
        res = np.log(np.nansum(series) / freq)
        return np.where(np.isinf(res), 0, res)

    # 获取barra_quote数据中的rstr数据
    def _get_rstr_barra(self, stocks, tdate, dates_d, params=(252,126,11,'000300.SH')):
        window, half_life, shift, index_code = params

        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life)
        idx = self._get_date_idx(tdate, dates_d)
        date_period = dates_d[idx-window-shift+1:idx+1]
        pct_chgs = self.pct_chg.T.loc[date_period, :]

        for i in range(1,shift+1):
            pct_chg = pct_chgs.iloc[i:i+window,:]
            stk_ret = pct_chg[stocks]
            bm_ret = pct_chg[index_code]
            excess_ret = np.log(1 + stk_ret).sub(np.log(1 + bm_ret), axis=0)
            excess_ret = excess_ret.mul(w, axis=0)
            rs = excess_ret.apply(np.nansum, axis=0)
            rs.name = f'rs_{i}'
            res = pd.concat([res, rs], axis=1)

        res['RSTR_barra'] = np.sum((res[f'rs_{i}'] for i in range(1,shift+1)), axis=0) / shift
        return res[['RSTR_barra']]

    def _get_barra_finance_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "mkt_cap_ard", "longdebttodebt", "other_equity_instruments_PRE", 
            "tot_equity", "tot_liab", "tot_assets", "pb_lf", 
            "pe_ttm", "pcf_ocf_ttm", "eps_diluted2", "orps"
            
            Default target barra_quote indicators:
            "MLEV_barra", "BLEV_barra", "DTOA_barra", "BTOP_barra", 
            "ETOP_barra", "CETOP_barra", "EGRO_barra", "SGRO_barra"
        """
        dat = pd.DataFrame(index=stocks)
        caldate = self.month_map[tdate]

        dat1 = self._get_leverage_barra(stocks, tdate, self.dates_d)
        dat2 = self._get_value_barra(stocks, caldate)
        #dat3 = self._get_growth_barra(stocks, caldate, params=(5,'y'))

        dat = reduce(self.concat_df, [dat, dat1, dat2, ])
        #dat = dat[self.barra_finance_target_indicators]
        return dat

    def _get_leverage_barra(self, stocks, tdate, dates):
        lst_tdate = self._get_date(tdate, -1, dates)
        caldate = self.month_map[tdate]
        dat = pd.DataFrame(index=stocks)
        try:
            long_term_debt = self.longdebttodebt_lyr_m.loc[stocks, caldate] * self.tot_liab_lyr_m.loc[stocks, caldate]
        except Exception:
            print(caldate, len(stocks))
            raise

        prefered_equity = self.other_equity_instruments_PRE_lyr_m.loc[stocks, caldate].fillna(0)

        dat['MLEV_barra'] = (prefered_equity + long_term_debt) / (self.mkt_cap_ard.loc[stocks, lst_tdate]) + 1
        dat['BLEV_barra'] = (self.tot_equity_lyr_m.loc[stocks, caldate] + long_term_debt) / (self.tot_equity_lyr_m.loc[stocks, caldate] - prefered_equity)
        dat['DTOA_barra'] = self.tot_liab_lyr_m.loc[stocks, caldate] / self.tot_assets_lyr_m.loc[stocks, caldate]
        return dat

    def _get_value_barra(self, stocks, caldate):
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat['BTOP_barra'] = 1 / self.pb_lf_m.loc[stocks, date]
        dat['ETOP_barra'] = 1 / self.pe_ttm_m.loc[stocks, date]
        dat['CETOP_barra'] = 1 / self.pcf_ocf_ttm_m.loc[stocks, date]
        return dat

    def _get_growth_barra(self, stocks, caldate, params=(5, 'y')):
        periods, freq = params
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        eps = self.eps_diluted2.loc[stocks,:]
        orps = self.orps.loc[stocks,:]
        dat['EGRO_barra'] = self._cal_growth_rate(eps, stocks, date, periods, freq)
        dat['SGRO_barra'] = self._cal_growth_rate(orps, stocks, date, periods, freq)
        return dat

    @staticmethod
    def _get_lyr_date(date):
        if date.month == 12:
            return date
        else:
            try:
                return pd.to_datetime(f'{date.year-1}-12-31')
            except:
                return pd.NaT

    def __cal_gr(self, series, lyr_rptdates, periods=5):
        lyr_date = lyr_rptdates[series.name]
        if pd.isna(lyr_date):
            return np.nan
        idx = self._get_date_idx(lyr_date, series.index)
        y = series.iloc[idx-periods+1:idx+1]
        x = pd.Series(range(1, len(y)+1), index=y.index)

        x_and_y = pd.concat([x,y], axis=1).dropna(how='any', axis=1)
        try:
            x, y = x_and_y.iloc[:, 0].values, x_and_y.iloc[:, 1].values
            _, coef = self.regress(x,y)
            return coef[0] / np.mean(y)
        except:
            return np.nan

    def _cal_growth_rate(self, ori_data, stocks, caldate, periods=5, freq='y'):
        try:
            current_rptdates = self.applied_rpt_date_M.loc[stocks, caldate]
        except Exception:
            print(stocks[:5])
            print(caldate)
            print(type(stocks), type(caldate))
            raise
        current_lyr_rptdates = current_rptdates.apply(self._get_lyr_date)
        #tdate = pd.to_datetime('2019-03-29'); self = z; caldate = self.month_map[tdate]
        #stocks = self._FactorProcess__get_stock_list(tdate); ori_data = self.current.loc[stocks,:]
        if ori_data.index.dtype == 'O':
            ori_data = ori_data.T
        ori_data = ori_data.groupby(pd.Grouper(freq=freq)).apply(lambda df: df.iloc[-1])
        res = ori_data.apply(self.__cal_gr, args=(current_lyr_rptdates, periods))
        return res

    @staticmethod
    def get_exponential_weights(window=12, half_life=6):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1]

    @staticmethod
    def winsorize(dat, n=5):
        dm = np.nanmedian(dat, axis=0)
        dm1 = np.nanmedian(np.abs(dat - dm), axis=0)
        if len(dat.shape) > 1:
            dm = np.repeat(dm.reshape(1,-1), dat.shape[0], axis=0)
            dm1 = np.repeat(dm1.reshape(1,-1), dat.shape[0], axis=0)
        dat = np.where(dat > dm + n * dm1, dm + n * dm1, 
              np.where(dat < dm - n * dm1, dm - n * dm1, dat))
        return dat

    @staticmethod
    def standardize(dat):
        dat_sta = (dat - np.nanmean(dat, axis=0)) / np.nanstd(dat, axis=0)
        return dat_sta

    @staticmethod
    def regress(X, y, intercept=True, weights=1, robust=False):
        if intercept:
            X = sm.add_constant(X)
        if robust:
            model = sm.RLM(y, X, weights=weights)
        else:
            model = sm.WLS(y, X, weights=weights)
        result = model.fit()
        params = result.params 
        return params[0], params[1:]

    @staticmethod
    def get_sma(df, n, m):
        try:
            sma = pd.ewma(df, com=n/m-1, adjust=False, ignore_na=True)
        except AttributeError:
            sma = df.ewm(com=n/m-1, min_periods=0, adjust=False, ignore_na=True).mean()
        return sma

    @staticmethod
    def get_ema(df, n):
        try:
            ema = pd.ewma(df, span=n, adjust=False, ignore_na=True)
        except AttributeError:
            ema = df.ewm(span=n, min_periods=0, adjust=False, ignore_na=True).mean()
        return ema

    # 获取日期索引
    def _get_daily_data(self, name, stocks, date, offset, datelist=None):
        dat = getattr(self, name, None)
        if dat is None:
            raise AttributeError("{} object has no attr: {}".format(self.__class__.__name__, name))

        dat = dat.loc[stocks, :].T
        if datelist is None:
            datelist = dat.index.tolist()
        idx = self._get_date_idx(date, datelist)
        start_idx, end_idx = max(idx-offset+1, 0), idx+1
        date_period = datelist[start_idx:end_idx]
        dat = dat.loc[date_period, :]
        return dat

    # 获取MACD数据
    def cal_MACD(self, stocks, date, params=(12,26,9)):
        n1, n2, m = params
        offset = max([n1,n2,m]) + 240
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        dif = self.get_ema(close, n1) - self.get_ema(close, n2)
        dea = self.get_ema(dif, m)
        macd = 2*(dif - dea)

        dif = dif.iloc[-1, :].T.values
        dea = dea.iloc[-1, :].T.values
        macd = macd.iloc[-1, :].T.values
        return dif, dea, macd

    # 获取PSY数据
    def cal_PSY(self, stocks, date, params=(20,)):
        m = params[0]
        offset = m + 1
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        con = (close > close.shift(1)).astype(int)
        psy = 100 * con.rolling(window=m).sum() / m

        return psy.iloc[-1, :].T.values

    # 获取RSI数据
    def cal_RSI(self, stocks, date, params=(20,)):
        n = params[0]
        offset = n + 1
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        delta = close - close.shift(1)
        tmp1 = delta.where(delta > 0, 0)
        tmp2 = delta.applymap(abs)
        rsi = 100 * self.get_sma(tmp1, n, 1) / self.get_sma(tmp2, n, 1)

        return rsi.iloc[-1, :].T.values

    # 获取BIAS数据
    def cal_BIAS(self, stocks, date, params=(20,)):
        n = params[0]
        offset = n
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        ma_close = close.rolling(window=n).mean()
        bias = 100 * (close - ma_close) / ma_close

        return bias.iloc[-1, :].T.values

    # 获取日期索引
    def _get_date_idx(self, date, datelist=None, ensurein=False):
        """获取日期在日期列表中的索引位置
        Args:
            date: 要查找的日期
            datelist: 日期列表,默认为None使用trade_days
            ensurein: 是否强制要求日期必须在列表中,默认False
        Returns:
            int: 日期在列表中的索引位置
        Raises:
            IndexError: 当ensurein=True且日期不在列表中时抛出
        """
        msg = """日期 {} 不在当前交易日列表中。如果确定该日期是交易日，
              请重置交易日列表为更长时间段或更高频率。"""
        date = pd.to_datetime(date)
        # 如果未提供日期列表,使用trade_days
        if datelist is None:
            datelist = self.trade_days
        try:
            # 对日期列表排序并查找目标日期的索引
            datelist = sorted(datelist)
            idx = datelist.index(date) # 获取索引位置
        except ValueError: # 日期不在列表中的处理
            if ensurein: # 如果要求日期必须在列表中,则抛出异常
                raise IndexError(msg.format(str(date)[:10]))
            # 将日期添加到列表中并排序
            dlist = list(datelist)
            dlist.append(date)
            dlist.sort()
            idx = dlist.index(date) # 获取添加后的索引位置
            # 如果日期是最后一个,说明超出了列表范围
            if idx == len(dlist)-1:
                raise IndexError(msg.format(str(date)[:10]))
            return idx - 1 # 返回日期前一个位置的索引
        return idx # 返回找到的索引位置

    def _get_date(self, date, offset=0, datelist=None):
        if datelist is None:
            datelist = self.trade_days
        try:
            idx = self._get_date_idx(date, datelist)
        except IndexError as e:
            print(e)
            idx = len(datelist) - 1
        finally:
            return datelist[idx+offset]

    def _get_period_d(self, date, offset=None, freq=None, datelist=None):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")

        conds = {}
        freq = freq.upper()
        if freq == "M":
            conds.update(months=-offset)
        elif freq == "Q":
            conds.update(months=-3*offset)
        elif freq == "Y":
            conds.update(years=-offset)
        else:
            freq = freq.lower()
            conds.update(freq=-offset)
        # 获取偏移后的日期
        start_date = pd.to_datetime(date) - pd.DateOffset(**conds)
        # 获取下一个月的第一天
        if start_date.month == 12:
            year = start_date.year + 1
            month = 1
        else:
            year = start_date.year
            month = start_date.month + 1
        day = 1

        sdate = datetime(year, month, day)

        if datelist is None:
            datelist = self.dates_d
        try:
            sindex = self._get_date_idx(sdate, datelist, ensurein=True)
            eindex = self._get_date_idx(date, datelist, ensurein=True)
            return datelist[sindex:eindex+1]
        except IndexError:
            return self._get_trade_days(sdate, date, "d")

            

    def _get_period(self, date, offset=-1, freq=None, datelist=None, resample=False):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")

        date = pd.to_datetime(date)
        if resample:
            if datelist:
                datelist = self._transfer_freq(datelist, freq)
            else:
                raise ValueError("Can resample on passed in datelist.")

        if freq is None or freq == self.freq:
            if datelist:
                end_idx = self._get_date_idx(date, datelist) + 1
            else:
                end_idx = self._get_date_idx(date) + 1
        else:
            if datelist:
                end_idx = self._get_date_idx(date, datelist) + 1
            else:
                msg = """Must pass in a datelist with freq={} since it is not conformed with default freq."""
                raise ValueError(msg.format(freq))
        start_idx = end_idx + offset
        return datelist[start_idx: end_idx]

    def _transfer_freq(self, daylist=None, freq='M'):
        if daylist is None:
            daylist = self.pct_chg_M.columns.tolist()
        freq = freq.upper()
        if freq == "M":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month)
        elif freq == "Q":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month and lst.month in (3,6,9,12))
        elif freq == "Y":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month and lst.month == 12)
        else:
            raise TypeError("Unsupported resample type {}.".format(freq))
        return list(res)


if __name__ == '__main__':
    gen = FactorGenerater()

    def create_factor_file(date):
        """创建因子文件的函数
        Args:
            date: 日期参数
            
        功能说明:
            1. 从日期获取文件名(取前10位字符)
            2. 尝试创建因子文件
            3. 如果文件已存在则打印提示信息
            4. 创建成功则打印完成信息
        """
        # 获取文件名(日期的前10位字符)
        sname = str(date)[:10]
        try:
            # 调用gen对象的create_factor_file方法创建因子文件
            gen.create_factor_file(date, os.path.join(WORK_PATH, "factors", f"{sname}.csv"))
        except FileAlreadyExistError:
            print(f"{sname}的数据已存在。") # 如果文件已存在,打印提示信息
        else:
            print(f"创建 {sname} 的数据完成。") # 创建成功,打印完成信息

    def main():
        # 确保factors目录存在
        factors_dir = os.path.join(WORK_PATH, "factors")
        if not os.path.exists(factors_dir):
            os.makedirs(factors_dir)
            print(f"创建目录: {factors_dir}")
        dates = [d for d in gen.month_map.keys()]
        s = pd.to_datetime('20170101')
        # s = pd.to_datetime('20241231')
        # e = pd.to_datetime('20241231')
        e = pd.to_datetime('20250429')
        dates = pd.Series(dates, index=dates)
        dates = dates[(dates>=s)&(dates<=e)]
        #串行
        for date in dates:
           create_factor_file(date)
        # #并行
        # function_list = [delayed(create_factor_file)(date) for date in dates]
        # Parallel(n_jobs=5, backend='multiprocessing')(function_list) #并行化处理

    main()
    # input()