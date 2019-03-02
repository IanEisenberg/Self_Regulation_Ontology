from kabuki.analyze import _parents_to_random_posterior_sample
import pymc.progressbar as pbar
import pandas as pd

def _post_pred_generate(bottom_node, samples=500, data=None, append_data=False):
    """Generate posterior predictive data from a single observed node."""
    datasets = []

    ##############################
    # Sample and generate stats
    for sample in range(samples):
        _parents_to_random_posterior_sample(bottom_node)
        # Generate data from bottom node
        sampled_data = bottom_node.random()
        if append_data and data is not None:
            #sampled_data = sampled_data.join(data.reset_index(), lsuffix='_sampled')
            sampled_data = sampled_data.merge(data, left_index=True, right_index=True, suffixes=('_sampled', ''))
        datasets.append(sampled_data)

    return datasets

def post_pred_gen(model, groupby=None, samples=500, append_data=False, progress_bar=True):
    """Run posterior predictive check on a model.

    :Arguments:
        model : kabuki.Hierarchical
            Kabuki model over which to compute the ppc on.

    :Optional:
        samples : int
            How many samples to generate for each node.
        groupby : list
            Alternative grouping of the data. If not supplied, uses splitting
            of the model (as provided by depends_on).
        append_data : bool (default=False)
            Whether to append the observed data of each node to the replicatons.
        progress_bar : bool (default=True)
            Display progress bar

    :Returns:
        Hierarchical pandas.DataFrame with multiple sampled RT data sets.
        1st level: wfpt node
        2nd level: posterior predictive sample
        3rd level: original data index

    :See also:
        post_pred_stats
    """
    results = {}

    # Progress bar
    if progress_bar:
        n_iter = len(model.get_observeds())
        bar = pbar.progress_bar(n_iter)
        bar_iter = 0
    else:
        print("Sampling...")

    if groupby is None:
        iter_data = ((name, model.data.ix[obs['node'].value.index]) for name, obs in model.iter_observeds())
    else:
        iter_data = model.data.groupby(groupby)

    for name, data in iter_data:
        node = model.get_data_nodes(data.index)
        
        #New addition: Reset index for non regression models
        if str(type(model)) == "<class 'hddm.models.hddm_info.HDDM'>":
            data = data.reset_index()

        if progress_bar:
            bar_iter += 1
            bar.update(bar_iter)

        if node is None or not hasattr(node, 'random'):
            continue # Skip

        ##############################
        # Sample and generate stats
        datasets = _post_pred_generate(node, samples=samples, data=data, append_data=append_data)
        results[name] = pd.concat(datasets, names=['sample'], keys=list(range(len(datasets))))
        
        #New addition: Convert results dict keys to single items for regression models with different conditions
    if list(results.keys())[0] != 'wfpt':
        if isinstance(list(results.keys())[0], str)==False:
            results={ '('+",".join(map(str,x))+')': results[x] for x in results.keys() }
            results={key.replace('.0','') : value for key, value in results.items()}
        else:
            results={ '('+x+')': results[x] for x in results.keys() }
            
    if progress_bar:
        bar_iter += 1
        bar.update(bar_iter)

    return pd.concat(results, names=['node'])