from typing import Optional, Union, Iterable
import numpy as np
import torch
from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam
import sys
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts")
from SPDNet.torch.layers import BatchNormSPD

import spdnets.modules as modules
import spdnets.batchnorm as bn
# from .layers import BiMap, LogEig, ReEig, BatchNormSPD



class TSMNet(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(subspacedims*(subspacedims+1)/2)
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        )

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )
        
        if self.domain_adaptation_:
            self.spdbnorm = bn.AdaMomDomainSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
        else:
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                          dispersion=self.bnorm_dispersion_, 
                                          learn_mean=False,learn_std=True,
                                          eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)

        self.spdnet = torch.nn.Sequential(
            modules.BiMap((1,self.spatial_filters_,subspacedims), dtype=self.spd_dtype_, device=self.spd_device_),
            modules.ReEig(threshold=1e-4),
        )
        self.logeig = torch.nn.Sequential(
            modules.LogEig(subspacedims),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim,self.nclasses_),
        )

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains):
        h = self.cnn(inputs[:,None,...])
        C = self.cov_pooling(h).to(self.spdnet[0].W)
        l = self.spdnet(C)
        l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        l = self.logeig(l)
        l = l.to(inputs)
        outputs = self.classifier(l)
        return outputs

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)



class SPDSMNet(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)
            
        self.bimap_layers = []
        self.spdbnorm_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            if self.domain_adaptation_:
                self.spdbnorm_layers.append(bn.AdaMomDomainSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            else:
                self.spdbnorm_layers.append(bn.AdaMomSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                            dispersion=self.bnorm_dispersion_, 
                                            learn_mean=False,learn_std=True,
                                            eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains):

        # print(inputs.size())
        l = inputs.clone()
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            l = self.spdbnorm_layers[i](l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
            # print(l.size())
            l = self.ReEig_layers[i](l)
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)
        return outputs

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)



class DSBNSPDBNNet(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bn_momentum=0.1,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation
        self.bn_momentum = bn_momentum

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)

        if self.domain_adaptation_:
            self.spdbnorm = bn.AdaMomDomainSPDBatchNorm((1,bimap_dims[0],bimap_dims[0]), batchdim=0, 
                            domains=self.domains_,
                            learn_mean=False,learn_std=True, 
                            dispersion=self.bnorm_dispersion_, 
                            eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
        else:
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,bimap_dims[0],bimap_dims[0]), batchdim=0, 
                                        dispersion=self.bnorm_dispersion_, 
                                        learn_mean=False,learn_std=True,
                                        eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
            
        self.bimap_layers = []
        self.bn_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            self.bn_layers.append(BatchNormSPD(momentum=self.bn_momentum, n=output_dim, dtype=self.spd_dtype_, device=self.spd_device_))
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains):

        # print(inputs.size())
        l = self.spdbnorm(inputs, domains)  if self.domain_adaptation_ else self.spdbnorm(l)
        l = l[:, None, :, :]
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            l = self.bn_layers[i](l)
            # print(l.size())
            l = self.ReEig_layers[i](l)
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)
        return outputs

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class DSBNSPDBNNet_Visu(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bn_momentum=0.1,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation
        self.bn_momentum = bn_momentum

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)

        if self.domain_adaptation_:
            self.spdbnorm = bn.AdaMomDomainSPDBatchNorm((1,bimap_dims[0],bimap_dims[0]), batchdim=0, 
                            domains=self.domains_,
                            learn_mean=False,learn_std=True, 
                            dispersion=self.bnorm_dispersion_, 
                            eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
        else:
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,bimap_dims[0],bimap_dims[0]), batchdim=0, 
                                        dispersion=self.bnorm_dispersion_, 
                                        learn_mean=False,learn_std=True,
                                        eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
            
        self.bimap_layers = []
        self.bn_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            self.bn_layers.append(BatchNormSPD(momentum=self.bn_momentum, n=output_dim, dtype=self.spd_dtype_, device=self.spd_device_))
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains, return_interbn=False):

        # print(inputs.size())
        out = []
        out += [inputs,] if return_interbn else []

        l = self.spdbnorm(inputs, domains)  if self.domain_adaptation_ else self.spdbnorm(l)
        out += [l,] if return_interbn else []

        l = l[:, None, :, :]
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            l = self.bn_layers[i](l)
            # print(l.size())
            l = self.ReEig_layers[i](l)
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)
        out = [outputs, None] if len(out) == 0 else [outputs, *out[::-1]]
        return out

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class SPDSMNet2(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)
            
        self.bimap_layers = []
        self.spdbnorm_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            if self.domain_adaptation_:
                self.spdbnorm_layers.append(bn.AdaMomDomainSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            else:
                self.spdbnorm_layers.append(bn.AdaMomSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                            dispersion=self.bnorm_dispersion_, 
                                            learn_mean=False,learn_std=True,
                                            eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains):

        # print(inputs.size())
        l = inputs.clone()
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            # print(l.size())
            l = self.ReEig_layers[i](l)
            l = self.spdbnorm_layers[i](l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)
        return outputs

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class SPDSMNet_visu(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)
            
        self.bimap_layers = []
        self.spdbnorm_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            if self.domain_adaptation_:
                self.spdbnorm_layers.append(bn.AdaMomDomainSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            else:
                self.spdbnorm_layers.append(bn.AdaMomSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                            dispersion=self.bnorm_dispersion_, 
                                            learn_mean=False,learn_std=True,
                                            eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains, return_interbn=False):

        # print(inputs.size())
        out = []
        l = inputs.clone()
        out += [l,] if return_interbn else []
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            l = self.spdbnorm_layers[i](l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
            # print(l.size())
            l = self.ReEig_layers[i](l)
            out += [l,] if return_interbn else []
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)

        out = [outputs, None] if len(out) == 0 else [outputs, *out[::-1]]
        return out

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class SPDSMNet_mnibatchbalance(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bimap_dims = [15],
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.bimap_dims = bimap_dims
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(self.bimap_dims[-1]*(self.bimap_dims[-1]+1)/2)
            
        self.bimap_layers = []
        self.spdbnorm_layers = []
        self.ReEig_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.bimap_layers.append(modules.BiMap((1, input_dim, output_dim), dtype=self.spd_dtype_, device=self.spd_device_))
            if self.domain_adaptation_:
                self.spdbnorm_layers.append(bn.BalancedAdaMomDomainSPDBatchNorm((1,output_dim,output_dim), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_))
            else:
                print("Aie AIe AIe ")
            self.ReEig_layers.append(modules.ReEig())
            input_dim = output_dim
        self.logeig = torch.nn.Sequential(
            modules.LogEig(self.bimap_dims[-1]),
            torch.nn.Flatten(start_dim=1),
        )
        lin_layer = torch.nn.Linear(tsdim, self.nclasses_, bias=False,dtype=torch.float64)#.double()
        torch.nn.init.xavier_uniform_(lin_layer.weight)
        self.lin = lin_layer

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.lin = self.lin.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        # self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, y, domains, return_interbn=False):

        # print(inputs.size())
        out = ()
        l = inputs.clone()
        out += (l,) if return_interbn else ()
        for i in range(len(self.bimap_dims[1:])):
            # print("device input",l.device)
            l = self.bimap_layers[i](l)
            # print(l.size())
            l = self.spdbnorm_layers[i](l,domains, y) if self.domain_adaptation_ else self.spdbnorm(l)
            # print(l.size())
            l = self.ReEig_layers[i](l)
            out += (l,) if return_interbn else ()
            # print(l.size())
        l = self.logeig(l)
        # l = l.to(inputs)
        
        outputs = self.lin(l)
        # h = self.cnn(inputs[:,None,...])
        # C = self.cov_pooling(h).to(self.spdnet[0].W)
        # l = self.spdnet(C)
        # l = self.spdbnorm(l,domains) if self.domain_adaptation_ else self.spdbnorm(l)
        # l = self.logeig(l)
        # l = l.to(inputs)
        # outputs = self.classifier(l)

        out = outputs if len(out) == 0 else (outputs, *out[::-1])
        return out

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation_:
            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)

            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d==du], d[d==du])

            for i in range(len(self.bimap_dims[1:])):
                self.spdbnorm_layers[i].set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)