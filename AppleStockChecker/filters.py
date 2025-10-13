import django_filters as df
from .models import PurchasingShopTimeAnalysis


class CSVNumberInFilter(df.BaseInFilter, df.NumberFilter):
    pass

class PurchasingShopTimeAnalysisFilter(df.FilterSet):
    start = df.IsoDateTimeFilter(method="filter_start")
    end = df.IsoDateTimeFilter(method="filter_end")
    def filter_start(self, qs, name, value):
        return qs.filter(Timestamp_Time__gte=value)

    def filter_end(self, qs, name, value):
        return qs.filter(Timestamp_Time__lte=value)
    # —— 基础时间过滤 —— #
    Timestamp_Time__gte = df.IsoDateTimeFilter(field_name="Timestamp_Time", lookup_expr="gte")
    Timestamp_Time__lte = df.IsoDateTimeFilter(field_name="Timestamp_Time", lookup_expr="lte")

    # —— 跨外键字段（join 过滤） —— #
    # iPhone 规格
    iphone__part_number = df.CharFilter(field_name="iphone__part_number", lookup_expr="iexact")
    iphone__jan = df.CharFilter(field_name="iphone__jan", lookup_expr="exact")
    iphone__model_name = df.CharFilter(field_name="iphone__model_name", lookup_expr="icontains")
    iphone__capacity_gb = df.NumberFilter(field_name="iphone__capacity_gb", lookup_expr="exact")
    iphone__color = df.CharFilter(field_name="iphone__color", lookup_expr="iexact")
    shop__in = CSVNumberInFilter(field_name="shop_id", lookup_expr="in")
    iphone__in = CSVNumberInFilter(field_name="iphone_id", lookup_expr="in")
    # Shop 信息
    shop__name = df.CharFilter(field_name="shop__name", lookup_expr="icontains")
    shop__address = df.CharFilter(field_name="shop__address", lookup_expr="icontains")

    class Meta:
        model = PurchasingShopTimeAnalysis
        fields = [
            "shop", "iphone", "Batch_ID", "Job_ID",
            # 下面这些是跨表筛选键
            "iphone__part_number", "iphone__jan", "iphone__model_name",
            "iphone__capacity_gb", "iphone__color",
            "shop__name",
            "Timestamp_Time__gte", "Timestamp_Time__lte",
        ]

